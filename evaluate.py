import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from early_stopping import EarlyStopping
import json
from utils_function import log_writer
from user_config import config_init
from model import InterF
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import util
from torch.utils.data import DataLoader
from entities import Dataset
import sampling
from input_reader import JsonInputReader



def evaluate(model, test_mode, optimizer, scheduler, epoch, args, early_stopping, test_dataset, dev_dataset, logger, writer):


    if test_mode == 'test':
        dataset = test_dataset
    elif test_mode == 'dev':
        dataset = dev_dataset
    # dataset = eval_data(test_mode, args.eval_batch_size, args.dataset, args.data_mode, args)
    dataset.switch_mode(Dataset.EVAL_MODE)
    data_loader = DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False,
                                 num_workers=4, collate_fn=sampling.collate_fn_padding)
        
    softmax = torch.nn.Softmax(dim=-1)
    model.eval()
    val_loss = 0.
    exist_loss = 0.
    span_loss = 0.
    sent_loss = 0.
    acc_sent = 0
    pre_sent = 0
    gold_sent = 0
    acc_span = 0
    pre_span = 0
    gold_span = 0
    sample_count = 0
    acc_exist = 0
    pre_exist = 0

    pre_span_strict, acc_span_strict, acc_sent_strict = 0, 0, 0
    with torch.no_grad():
        for dno, data in enumerate(tqdm(data_loader)):

            data = util.to_device(data, args.device)


            if args.model_mode == "InterF":
                encoding = data['encodings']
                attention_mask = data['context_masks'].float()
                span_pool_mask = data['entity_masks']
                left_context_mask = data['left_masks']
                right_context_mask = data['right_masks']
                gold_iv = data['sentence_types']
                gold_spans = data['entity_types']
                span_mask = data['entity_sample_masks']
                spans = data['entity_spans']
                entity_width = data['entity_sizes']

                gold_exist, exist_mask = util.create_exist(encoding.shape[0], encoding.shape[1], attention_mask, spans, gold_spans)

                val_loss, span_select, span_prob, sent_prob, loss_span, loss_iv, loss_exist, _, _, _, _ = model(input_ids=encoding, attention_mask=attention_mask, span_pool_mask=span_pool_mask, gold_exist=gold_exist, gold_span=gold_spans, gold_iv=gold_iv, exist_mask=exist_mask, span_mask=span_mask, left_context_mask=left_context_mask, right_context_mask=right_context_mask, span_width=entity_width, spans=spans, is_training=False)
                span_prob = torch.max(span_prob, dim=-1)[1]
                sent_prob[sent_prob >= 0.5] = 1
                sent_prob[sent_prob != 1] = 0
                span_select = torch.max(span_select, dim=-1)[1]
                span_loss += loss_span.item()
                exist_loss += loss_exist.item()
                sent_loss += loss_iv.item()
                for no, (span_matrix, g_span, g_sent, p_span, p_sent, i_span) in enumerate(zip(span_select, gold_spans, gold_iv, span_prob, sent_prob, spans)):
                    matrix = span_matrix.cpu().numpy()
                    index = np.argwhere(matrix == 1)
                    for idx in index:
                        if exist_mask[no][idx[0]][idx[1]].item() == 1:
                            pre_exist += 1
                        idx[1] += 1
                    for span_index, span, span_type, iv, span_iv in zip(i_span, g_span, p_span, g_sent, p_sent):
                        if [span_index[0].item(), span_index[1].item()] in index and span.item() != 0 and exist_mask[no][span_index[0].item()][span_index[1].item() - 1].item() == 1:
                            acc_exist += 1
                        
                        # consider span_index
                        # if [span_index[0].item(), span_index[1].item()] in index and span_type.item() != 0:
                        #     pre_span_strict += 1
                        #     if span_type.item() == span.item():
                        #         acc_span_strict += 1
                        #         if iv.item() == span_iv.item():
                        #             acc_sent_strict += 1
                        
                        if span_type.item() != 0:
                            pre_span += 1
                            if span_type.item() == span.item():
                                acc_span += 1
                                if iv.item() == span_iv.item():
                                    acc_sent += 1
                

                gold_sent += len(dataset.documents[dno].entities)
                gold_span += len(dataset.documents[dno].entities)


            elif args.model_mode == "InterF_ent":
                encoding = data['encodings']
                attention_mask = data['context_masks'].float()
                span_pool_mask = data['entity_masks']
                gold_spans = data['entity_types']
                span_mask = data['entity_sample_masks']
                spans = data['entity_spans']
                entity_width = data['entity_sizes']

                gold_exist, exist_mask = util.create_exist(encoding.shape[0], encoding.shape[1], attention_mask, spans, gold_spans)

                val_loss, span_select, span_prob, loss_span, loss_exist = model(input_ids=encoding, attention_mask=attention_mask, span_pool_mask=span_pool_mask, gold_exist=gold_exist, gold_span=gold_spans, exist_mask=exist_mask, span_mask=span_mask, span_width=entity_width, spans=spans, is_training=False)
                span_prob = torch.max(span_prob, dim=-1)[1]
                span_select = torch.max(span_select, dim=-1)[1]
                span_loss += loss_span.item()
                exist_loss += loss_exist.item()

                for no, (span_matrix, g_span, p_span, i_span) in enumerate(zip(span_select, gold_spans, span_prob, spans)):
                    matrix = span_matrix.cpu().numpy()
                    index = np.argwhere(matrix == 1)
                    for idx in index:
                        if exist_mask[no][idx[0]][idx[1]].item() == 1:
                            pre_exist += 1
                        idx[1] += 1
                    for span_index, span, span_type in zip(i_span, g_span, p_span):
                        if [span_index[0].item(), span_index[1].item()] in index and span.item() != 0 and exist_mask[no][span_index[0].item()][span_index[1].item() - 1].item() == 1:
                            acc_exist += 1                        
                        if span_type.item() != 0:
                            pre_span += 1
                            if span_type.item() == span.item():
                                acc_span += 1

                gold_span += len(dataset.documents[dno].entities)


        if args.model_mode == "InterF_ent":
            recall_exist, precision_exist, f_measure_exist = calculate_rpf(logger,acc_exist, pre_exist, gold_span, 'exist')
            recall_span, precision_span, f_measure_span = calculate_rpf(logger,acc_span, pre_span, gold_span, 'span')
            save_log(logger, test_mode + '-Exist', recall_exist, precision_exist, f_measure_exist, exist_loss)
            save_log(logger, test_mode + '-Entity', recall_span, precision_span, f_measure_span, span_loss)
            f_measure_sent = 0
        elif args.model_mode == "InterF":
            recall_exist, precision_exist, f_measure_exist = calculate_rpf(logger,acc_exist, pre_exist, gold_span, 'exist')
            recall_span, precision_span, f_measure_span = calculate_rpf(logger,acc_span, pre_span, gold_span, 'span')
            recall_sent, precision_sent, f_measure_sent = calculate_rpf(logger,acc_sent, pre_span, gold_span, 'sent')
            save_log(logger, test_mode + '-Exist', recall_exist, precision_exist, f_measure_exist, exist_loss)
            save_log(logger, test_mode + '-Entity', recall_span, precision_span, f_measure_span, span_loss)
            save_log(logger, test_mode + '-Sentence', recall_sent, precision_sent, f_measure_sent, sent_loss)
            writer.add_scalar(tag='Exist-F1/' + test_mode, scalar_value=f_measure_exist, global_step=epoch)
            writer.add_scalar(tag='Entity-F1/' + test_mode, scalar_value=f_measure_span, global_step=epoch)
            writer.add_scalar(tag='Sentence-F1/' + test_mode, scalar_value=f_measure_sent, global_step=epoch)
        
        writer.add_scalar(tag='Loss/' + test_mode, scalar_value=val_loss, global_step=epoch)
        
        
        # save model
        if test_mode == 'dev':
            print("total loss: %f" % val_loss )
            early_stopping(val_loss, model, optimizer, scheduler, epoch, args)
           
        return f_measure_span, f_measure_sent



def one_batch(tensor_list):
    for no in range(len(tensor_list)):
        tensor_list[no] = tensor_list[no].unsqueeze(0).to(tensor_list[no].device)
    return tuple(tensor_list)



def calculate_rpf(logger, acc, pre, gold, label):
    # avoid div 0 error
    recall = acc / gold
    if pre > 0:
        precision = acc / pre
        if recall > 0:
            f_measure = 2 * acc / (pre + gold)
        else:
            f_measure = 0
    else:
        precision = 0
        f_measure = 0

    logger.info(label + ": acc label: %d, gold label: %d, pre label: %d" % (acc, gold, pre))  
    print(label + ": acc label: %d, gold label: %d, pre label: %d" % (acc, gold, pre))

    # calculate recall/precision/f
    recall = round(recall * 100, 2)
    precision = round(precision * 100, 2)
    f_measure = round(f_measure * 100, 2)

    return recall, precision, f_measure


def save_log(logger, name, recall, precision, f_measure, loss=None):
    if loss is not None:
        logger.info(name + ': Loss - %f, Recall - %f ; Precision - %f ; F-measure - %f' % (loss, recall, precision, f_measure))
        print(name + ': Loss - %f, Recall - %f ; Precision - %f ; F-measure - %f' % (loss, recall, precision, f_measure))
    else:
        logger.info(name + ': Recall - %f ; Precision - %f ; F-measure - %f' % (recall, precision, f_measure))
        print(name + ': Recall - %f ; Precision - %f ; F-measure - %f' % (recall, precision, f_measure))