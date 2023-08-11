import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from entities import Dataset
import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoConfig
from torch.optim import AdamW, Adam
from utils_function import log_writer, data_encode, save_model, set_seed, data_to_device
from input_reader import JsonInputReader, JsonPredictionInputReader
from user_config import config_init
from model import InterF, InterF_ent
from early_stopping import EarlyStopping
from evaluate import evaluate
from torch.utils.data import DataLoader
# from predict import predict
from tqdm import tqdm
import sampling
import util
import time
import json
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter



# train
def train(args):

    logger = log_writer(args.log_path + args.data_mode + str(args.random_seed) + '_log.log')

    logger.info('seed value: %s' % args.random_seed)


    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    bert_config = AutoConfig.from_pretrained(args.bert_model)
    writer = SummaryWriter(log_dir="./runs/", flush_secs=120)

    # initialize model and data
    
    type_path = os.path.join(args.dataset, args.data_mode + '/', 'type.json')
    train_path = os.path.join(args.dataset, args.data_mode + '/', 'train.json')
    dev_path = os.path.join(args.dataset, args.data_mode + '/', 'dev.json')
    test_path = os.path.join(args.dataset, args.data_mode + '/', 'test.json')

    input_reader = JsonInputReader(type_path, tokenizer, args.sample_count, args.max_span_len, logger)

    
    type_file = open(type_path, 'r', encoding='utf-8')
    entity_type = json.loads(type_file.read())
    entity_to_idx = {}
    for no, (k ,v) in enumerate(entity_type['entities'].items()):
        entity_to_idx[k] = no + 1

    entity_to_idx['none'] = 0
    args.num_labels = len(entity_to_idx)

    
    # load dataset
    train_dataset = input_reader.read(train_path, 'train')
    dev_dataset = input_reader.read(dev_path, 'valid')
    test_dataset = input_reader.read(test_path, 'test')
    
    # record dataset info
    log_datasets(logger, input_reader)

    train_sample_count = train_dataset.document_count
    updates_epoch = train_sample_count // args.batch_size
    updates_total = updates_epoch * args.epoch

    if args.model_mode == 'InterF':
        model = InterF.from_pretrained(args.bert_model, config=bert_config, user_config=args)
    elif args.model_mode == 'InterF_v5_ent':
        model = InterF_ent.from_pretrained(args.bert_model, config=bert_config, user_config=args)
    else:
        raise Exception('model mode error')
    
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    # use different learning rate for different part of the model
    no_weight_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    bert_params = list(model.bert.parameters())
    bert_param_ids = list(map(id, bert_params))
    no_weight_decay_params = [x[1] for x in filter(
        lambda name_w: any(nwd in name_w[0] for nwd in no_weight_decay), model.named_parameters())]
    no_weight_decay_param_ids = list(map(id, [x for x in no_weight_decay_params]))

    bert_base_params = filter(lambda p: id(p) in bert_param_ids and id(p) not in no_weight_decay_param_ids,
                                  model.parameters())
    bert_no_weight_decay_params = filter(lambda p: id(p) in bert_param_ids and id(p) in no_weight_decay_param_ids,
                                            model.parameters())
    core_param_ids = [id(model.core)]
    base_no_weight_decay_params = filter(
            lambda p: id(p) not in bert_param_ids and id(p) in no_weight_decay_param_ids and id(p) not in core_param_ids,
            model.parameters())
    base_params = filter(lambda p: id(p) not in bert_param_ids and id(p) not in no_weight_decay_param_ids and id(p) not in core_param_ids,
                            model.parameters())
    params = [{"params": bert_base_params, "lr": args.learning_rate},
              {"params": bert_no_weight_decay_params, "lr": args.learning_rate, "weight_decay": 0.0},
              {"params": base_no_weight_decay_params, "lr": args.learning_rate, "weight_decay": 0.0},
              {"params": base_params, "lr": args.learning_rate},
              {"params": model.core, "lr": args.core_learning_rate}]
    

    optimizer = AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.lr_warmup *
                                                             updates_total,
                                                             num_training_steps=updates_total)
    

    # load existing model
    start_epoch = -1
    if args.train_mode == 'C':
        checkpoint = torch.load(os.path.join(args.checkpoint_path, args.data_mode + '_' + args.model_mode + '_' + args.context + '.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        early_stopping = EarlyStopping(args.checkpoint_path, args.model_path, val_loss_min=checkpoint['val_loss_min'], 
                                      counter=checkpoint['counter'], delta=checkpoint['delta'], best_score=checkpoint['best_score'])
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, args.model_path)
    
    
    

    f_measure_max = 0.


    logger.info('--------Model: %s--Data: %s--Device: %s--lr: %s--core_lr: %s--------' % (args.model_mode, args.data_mode, args.device, args.learning_rate, args.core_learning_rate))

    print('--------Model: %s--Data: %s--Device: %s--lr: %s--core_lr: %s-------' % (args.model_mode, args.data_mode, args.device, args.learning_rate, args.core_learning_rate))


    # train model
    for epoch in range(start_epoch + 1, args.epoch):
        logger.info('--------Epoch: %d--------' % epoch)
        train_loss = 0
        span_total_loss = 0
        sent_total_loss = 0
        index_total_loss = 0
        
        train_dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False,
                                 num_workers=4, collate_fn=sampling.collate_fn_padding)
        
        
        total = train_dataset.document_count // args.batch_size

        for data in tqdm(data_loader, total=total, desc='Train epoch %s' % epoch):
            model.train()
            model.zero_grad()
            
            # grad_optimizer.zero_grad()
            data = util.to_device(data, args.device)

            if args.model_mode == 'InterF':
                encoding = data['encodings']
                attention_mask = data['context_masks'].float()
                span_pool_mask = data['entity_masks']
                left_context_mask = data['left_masks']
                right_context_mask = data['right_masks']
                gold_iv = data['sentence_types']
                gold_span = data['entity_types']
                span_mask = data['entity_sample_masks']
                spans = data['entity_spans']
                entity_width = data['entity_sizes']

                gold_exist, exist_mask = util.create_exist(encoding.shape[0], encoding.shape[1], attention_mask, spans, gold_span)


                loss, span_logits, sent_logits, loss_span, loss_iv, loss_exist = model(input_ids=encoding, attention_mask=attention_mask, span_pool_mask=span_pool_mask, gold_exist=gold_exist, gold_span=gold_span, gold_iv=gold_iv, exist_mask=exist_mask, span_mask=span_mask, left_context_mask=left_context_mask, right_context_mask=right_context_mask, span_width=entity_width, spans=spans, is_training=True)
                span_total_loss += loss_span.item()
                sent_total_loss += loss_iv.item()
                index_total_loss += loss_exist.item()
            
            elif args.model_mode == 'InterF_ent':
                encoding = data['encodings']
                attention_mask = data['context_masks'].float()
                span_pool_mask = data['entity_masks']
                gold_span = data['entity_types']
                span_mask = data['entity_sample_masks']
                spans = data['entity_spans']
                entity_width = data['entity_sizes']
                gold_exist, exist_mask = util.create_exist(encoding.shape[0], encoding.shape[1], attention_mask, spans, gold_span)

                loss, span_logits, loss_span, loss_exist = model(input_ids=encoding, attention_mask=attention_mask, span_pool_mask=span_pool_mask, gold_exist=gold_exist, gold_span=gold_span, exist_mask=exist_mask, span_mask=span_mask, span_width=entity_width, spans=spans, is_training=True)
            

            train_loss += loss.item()
            
            # grad_loss.additional_forward_and_backward(model, grad_optimizer)
            loss.backward() 
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        

        print('Epoch: %d Finished' % epoch)
        print('train-Entity: Loss - %f' % train_loss)
        writer.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
        f_measure_span, f_measure_sent = evaluate(model, 'dev', optimizer, scheduler, epoch, args, early_stopping, test_dataset, dev_dataset, logger, writer)
        
        # choose the best model
        if args.save_mode == 'sent' and args.model_mode != "InterF_ent":
            if f_measure_sent > f_measure_max:
                f_measure_max = f_measure_sent
                save_model(model, optimizer, epoch, scheduler, args)
                f_measure_span, f_measure_sent = evaluate(model, 'test', optimizer, scheduler, epoch, args, early_stopping, test_dataset, dev_dataset, logger, writer)
        elif args.save_mode == 'span' or args.model_mode == "InterF_ent":
            if f_measure_span > f_measure_max:
                f_measure_max = f_measure_span
                save_model(model, optimizer, epoch, scheduler, args)
                f_measure_span, f_measure_sent = evaluate(model, 'test', optimizer, scheduler, epoch, args, early_stopping, test_dataset, dev_dataset, logger, writer)


# log the dataset information
def log_datasets(logger, input_reader):
        logger.info("Entity type count: %s" % input_reader.entity_type_count)

        logger.info("Entities:")
        for e in input_reader.entity_types.values():
            logger.info(e.verbose_name + '=' + str(e.index))
        
        for s in input_reader.sentence_types.values():
            logger.info(s.verbose_name + '=' + str(s.index))

        for k, d in input_reader.datasets.items():
            logger.info('Dataset: %s' % k)
            logger.info("Document count: %s" % d.document_count)
            logger.info("Entity count: %s" % d.entity_count)


# log the evaluation results
def save_log(logger, name, recall, precision, f_measure, loss=None):
    if loss is not None:
        logger.info(name + ': Loss - %f, Recall - %f ; Precision - %f ; F-measure - %f' % (loss, recall, precision, f_measure))
        print(name + ': Loss - %f, Recall - %f ; Precision - %f ; F-measure - %f' % (loss, recall, precision, f_measure))
    else:
        logger.info(name + ': Recall - %f ; Precision - %f ; F-measure - %f' % (recall, precision, f_measure))
        print(name + ': Recall - %f ; Precision - %f ; F-measure - %f' % (recall, precision, f_measure))


# run
if __name__ == '__main__':
    args = config_init()
    
    # set seed
    if args.random_seed is not None:
        set_seed(args.random_seed)
    else:
        seed = int(time.time())   
        args.random_seed = seed 
        set_seed(seed)

    train(args)
