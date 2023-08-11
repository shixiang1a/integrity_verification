import os
from turtle import Turtle
import torch
from data_loader import predict_data, eval_data
import json
from user_config import config_init
from transformers import AutoTokenizer, AutoConfig
from model import InterF
from evaluate import evaluate_wot
from utils_function import log_writer, data_encode
import torch
import transformers
from tqdm import tqdm
from torch.utils.data import DataLoader
from entities import Dataset, TokenSpan
import sampling
from input_reader import JsonInputReader
import util
import pandas as pd


def predict(model, tokenizer, entity_to_idx, type_path, args):

    logger = log_writer(args.log_path + args.data_mode + '_log.log')
    test_path = os.path.join(args.dataset, args.data_mode + '/', 'test.json')
    input_reader = JsonInputReader(type_path, tokenizer, args.sample_count, args.max_span_len, logger)
    dataset = input_reader.read(test_path, 'test')
    dataset.switch_mode(Dataset.EVAL_MODE)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False,
                                 num_workers=4, collate_fn=sampling.collate_fn_padding)

    model.eval()
    predict_list = []
    idx_to_entity = {v: k for k, v in entity_to_idx.items()}

    org_file = open(test_path, 'r', encoding='utf-8')
    org_res = org_file.readlines()

    documents = dataset.documents
    # file = open(args.result_path + args.model_mode + '.json', 'a', encoding='utf-8')

    results = []

    with torch.no_grad():
        for idx, (data, res) in enumerate(zip(tqdm(data_loader), org_res)):
            data = util.to_device(data, args.device)

            doc = documents[idx]
            org_data = json.loads(res)

            start_loc_map = {}
            end_loc_map = {}
            new_tokens = ['[CLS]']
            for loc, t in enumerate(org_data['sentence']):
                start_loc_map[len(new_tokens)] = loc
                s = tokenizer.tokenize(t)
                if len(s) > 0:
                    new_tokens += s
                else:
                    new_tokens += [t]
                end_loc_map[len(new_tokens)] = loc
            new_tokens += ['[SEP]']

            
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

                _, span_select, span_prob, sent_prob, _, _, _, atten_left, atten_right, left_prob, right_prob = model(input_ids=encoding, attention_mask=attention_mask, span_pool_mask=span_pool_mask, gold_exist=gold_exist, gold_span=gold_spans, gold_iv=gold_iv, exist_mask=exist_mask, span_mask=span_mask, left_context_mask=left_context_mask, right_context_mask=right_context_mask, span_width=entity_width, spans=spans, is_training=False)
                span_prob = torch.max(span_prob, dim=-1)[1]
                sent_prob[sent_prob >= 0.5] = 1
                sent_prob[sent_prob != 1] = 0

                for span_pre, sent_pre, mask, index, aleft, aright, lp, rp in zip(span_prob, sent_prob, span_mask, spans, atten_left, atten_right, left_prob, right_prob):
                    entities_list = []
                    pred_true = False
                    for no, (p, entity_span) in enumerate(zip(span_pre, index)):
                        if mask[no].item() == 1 and p.item() != 0:
                            if sent_pre[no].item() == 1:
                                iv = True
                            else:
                                iv = False
                            try:
                                entities_list.append([entity_span[0].item(), entity_span[1].item(), idx_to_entity[p.item()], iv])
                            except:
                                entities_list.append([-1, -1, idx_to_entity[p.item()], iv])

                    
                    for n in entities_list:
                        n[0] = start_loc_map[n[0]]
                        n[1] = end_loc_map[n[1]]
                    
                    acc_ent = [e for e in entities_list if e in org_data['ner']]
                    if len(entities_list) == len(org_data['ner']) == len(acc_ent):
                        pred_true = True

                    results.append(
                        {
                            "pred": pred_true,
                            "sentence": org_data['sentence'],
                            "predicted_ner": sorted(entities_list),
                            "gold_ner": org_data['ner'],
                        }
                    )

    
    df = pd.DataFrame(results)
    df.to_excel('/result/pred_sym.xlsx')


def direct_predict(args):
    type_path = os.path.join(args.dataset, args.data_mode + '/', 'type.json')
    type_file = open(type_path, 'r', encoding='utf-8')
    entity_type = json.loads(type_file.read())
    entity_to_idx = {}
    for no, (k ,v) in enumerate(entity_type['entities'].items()):
        entity_to_idx[k] = no + 1

    entity_to_idx['none'] = 0
    args.num_labels = len(entity_to_idx)

    bert_config = AutoConfig.from_pretrained(args.bert_model)
    
    model = InterF.from_pretrained(args.bert_model, config=bert_config, user_config=args)

    model.to(args.device)
    # model = torch.nn.DataParallel(model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    if args.use_es:
        checkpoint = torch.load(os.path.join(args.checkpoint_path, args.data_mode + '_' + args.model_mode + '_' + str(args.random_seed) + str(args.max_span_len) + 'es.pth'))
    else:
        checkpoint = torch.load('/checkpoint/INTER_SYM_InterF_v5_best.pth')
    model.load_state_dict(checkpoint['model'])

    predict(model, tokenizer, entity_to_idx, type_path, args)


if __name__ == '__main__':
    args = config_init()
    direct_predict(args)