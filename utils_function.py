from multiprocessing import pool
import os
import logging
import time
import torch
import os
import random
import numpy as np
from tqdm import tqdm
import time
# write log
def log_writer(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file_path, mode='a')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


# attention mask
def attention_masks(input_ids):
    atten_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]  # PAD: 0; 否则: 1
        atten_masks.append(seq_mask)
    return atten_masks




# cut sentence and entity that exceeds max-length
def cut_sentence_span(entity, token):
        
    if len(token) > 511:
        new_token = token[:511]
        new_entity = []
        for e in entity:
            if e[0] < 511 and e[1] < 511:
                new_entity.append(e)
    else:
        new_entity = entity
        new_token = token
    
    return new_entity, new_token
    

def extend_tensor(tensor, extended_shape, fill=0):
    tensor_shape = tensor.shape

    extended_tensor = torch.zeros(extended_shape, dtype=tensor.dtype).to(tensor.device)
    extended_tensor = extended_tensor.fill_(fill)

    if len(tensor_shape) == 1:
        extended_tensor[:tensor_shape[0]] = tensor
    elif len(tensor_shape) == 2:
        extended_tensor[:tensor_shape[0], :tensor_shape[1]] = tensor
    elif len(tensor_shape) == 3:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2]] = tensor
    elif len(tensor_shape) == 4:
        extended_tensor[:tensor_shape[0], :tensor_shape[1], :tensor_shape[2], :tensor_shape[3]] = tensor

    return extended_tensor



def sentence_span_generation(sent_len, input_len, max_span_len, entity, entity_to_idx, sample_count, is_sample):
    ner_loc = []
    ner_label = []
    ner_integrity = []
    for e in entity:
        ner_loc.append([e[0], e[1]])
        ner_label.append(e[2])
        ner_integrity.append(e[3])

    pos_span, pos_span_label, neg_span, neg_span_label = [], [], [], []
    pos_sent_mask, pos_sent_label, neg_sent_mask, neg_sent_label = [], [], [], []
    pos_pool_mask, pos_left_mask, pos_right_mask = [], [], []
    neg_pool_mask, neg_left_mask, neg_right_mask = [], [], []


    for i in range(1, sent_len + 1):
        for j in range(i + 1, min(sent_len + 2, i + max_span_len + 1)):
            if [i, j] in ner_loc:
                pos_span.append([i, j, entity_to_idx[ner_label[ner_loc.index([i, j])]], j - i])
                pos_span_label.append(entity_to_idx[ner_label[ner_loc.index([i, j])]])
                pos_sent_label.append(ner_integrity[ner_loc.index([i, j])])
                pos_sent_mask.append(1)
                pos_pool_mask.append([1 if i<=k<j else 0 for k in range(input_len)])
                pos_left_mask.append([1 if 0<k<j else 0 for k in range(input_len)])
                pos_right_mask.append([1 if i<=k<sent_len + 1 else 0 for k in range(input_len)])
            else:
                neg_span.append([i, j, 0, j - i])
                neg_span_label.append(0)
                neg_sent_mask.append(0)
                neg_sent_label.append(0.)
                neg_pool_mask.append([1 if i<=k<j else 0 for k in range(input_len)])
                neg_left_mask.append([1 if 0<k<j else 0 for k in range(input_len)])
                neg_right_mask.append([1 if i<=k<sent_len + 1 else 0 for k in range(input_len)])
    
    if is_sample:
        neg_sample = random.sample(list(zip(neg_span, neg_span_label, neg_sent_mask, neg_sent_label, neg_pool_mask, neg_left_mask, neg_right_mask)), min(len(neg_span), sample_count))
        neg_sample_span, neg_sample_span_label, neg_sample_sent_mask, neg_sample_sent_label, neg_sample_pool_mask, neg_sample_left_mask, neg_sample_right_mask = zip(*neg_sample) if neg_sample else ([], [])
    
        span = pos_span + list(neg_sample_span)
        span_label = pos_span_label + list(neg_sample_span_label)
        sent_mask = pos_sent_mask + list(neg_sample_sent_mask)
        sent_label = pos_sent_label + list(neg_sample_sent_label)
        pool_mask = pos_pool_mask + list(neg_sample_pool_mask)
        left_mask = pos_left_mask + list(neg_sample_left_mask)
        right_mask = pos_right_mask + list(neg_sample_right_mask)
    
    else:
        span = pos_span + neg_span
        span_label = pos_span_label + neg_span_label
        sent_mask = pos_sent_mask + neg_sent_mask
        sent_label = pos_sent_label + neg_sent_label
        pool_mask = pos_pool_mask + neg_pool_mask
        left_mask = pos_left_mask + neg_left_mask
        right_mask = pos_right_mask + neg_right_mask
    
    return span, span_label, sent_mask, sent_label, pool_mask, left_mask, right_mask



def sentence_span_encode_v2(batch_data, entity_to_idx, tokenizer, max_span_len, sample_count, mode, train_dataloader, args):
    input_ids = []
    new_batch_entities = []
    batch_entities = batch_data['entity']
    batch_sentences = batch_data['sentence']
    org_sent_len = []
    for no, entity in enumerate(batch_entities):
        sentence = batch_sentences[no]
        
        start2idx = []
        end2idx = []
        bert_tokens = []
        bert_tokens += [tokenizer.cls_token]

        for token in sentence:
            start2idx.append(len(bert_tokens))
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) == 0:
                sub_tokens = token
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens))
        
        org_sent_len.append(len(bert_tokens) - 1)
        
        
        for e in entity:
            e[0] = start2idx[e[0]]
            e[1] = end2idx[e[1] - 1]
        
        # cut sentence and entity that exceeds max-length
        new_entity, new_bert_tokens = cut_sentence_span(entity, bert_tokens)

        new_bert_tokens += [tokenizer.sep_token]
        
        new_batch_entities.append(new_entity)

        input_id = tokenizer.convert_tokens_to_ids(new_bert_tokens)
        input_ids.append(input_id) 
    
    max_len = max(map(len, input_ids))

    new_input_ids = []
    batch_span_label = []
    batch_span_mask = []
    batch_span = []
    batch_sent_mask = []
    batch_sent_label = []
    batch_span_pool_mask = []
    batch_left_context_mask = [] 
    batch_right_context_mask = []
    max_span_count = 0


    for no, input_id in enumerate(input_ids):
        new_input_id = input_id + [tokenizer.pad_token_id for _ in range(max_len - len(input_id))]
        new_input_ids.append(new_input_id)
        if mode == 'train':
            span, span_label, sent_mask, sent_label, pool_mask, left_mask, right_mask = sentence_span_generation(org_sent_len[no], len(new_input_id), max_span_len, new_batch_entities[no], entity_to_idx, sample_count, is_sample=True)
        else:
            span, span_label, sent_mask, sent_label, pool_mask, left_mask, right_mask = sentence_span_generation(org_sent_len[no], len(new_input_id), max_span_len, new_batch_entities[no], entity_to_idx, sample_count, is_sample=False)
        batch_span_label.append(span_label)
        batch_span_mask.append([1 for _ in range(len(span))])
        batch_span.append(span)
        batch_sent_mask.append(sent_mask)
        batch_sent_label.append(sent_label)
        max_span_count = max(max_span_count, len(span))
        batch_span_pool_mask.append(pool_mask)
        batch_left_context_mask.append(left_mask)
        batch_right_context_mask.append(right_mask)

    
    for no, _ in enumerate(zip(batch_span, batch_span_mask, batch_span_label, batch_sent_label, batch_sent_mask, new_input_ids)):
        b_span = []
        b_span_mask = []
        b_span_label = []
        b_sent_label = []
        b_sent_mask = []
        b_pool_mask = []
        b_left_mask = []
        b_right_mask = []

        for _ in range(max_span_count - len(batch_span[no])):
            b_span.append([0, 1, 0, 1])
            b_span_mask.append(0)
            b_span_label.append(0)
            b_sent_label.append(0)
            b_sent_mask.append(0)
            b_pool_mask.append([1 for _ in range(len(new_input_ids[no]))])
            b_left_mask.append([1 for _ in range(len(new_input_ids[no]))])
            b_right_mask.append([1 for _ in range(len(new_input_ids[no]))])
        
        batch_span[no] += b_span
        batch_span_mask[no] += b_span_mask
        batch_span_label[no] += b_span_label
        batch_sent_label[no] += b_sent_label
        batch_sent_mask[no] += b_sent_mask
        batch_span_pool_mask[no] += b_pool_mask
        batch_left_context_mask[no] += b_left_mask
        batch_right_context_mask[no] += b_right_mask
    


    return new_input_ids, batch_sent_label, batch_sent_mask, batch_span_label, batch_span_mask, batch_span, batch_span_pool_mask, batch_left_context_mask, batch_right_context_mask



def sentence_span_encode_v4(batch_data, entity_to_idx, tokenizer, sample_count, train_mode):
    input_ids, org_sent_len, batch_start2idx, batch_end2idx = [], [], [], []
    batch_entities = batch_data['entity']
    batch_sentences = batch_data['sentence']
    for no, entity in enumerate(batch_entities): 
        sentence = batch_sentences[no]  
        start2idx, end2idx, bert_tokens = [], [], []
        bert_tokens += [tokenizer.cls_token]

        for token in sentence:
            start2idx.append(len(bert_tokens))
            sub_tokens = tokenizer.tokenize(token)
            if len(sub_tokens) == 0:
                sub_tokens = token
            bert_tokens += sub_tokens
            end2idx.append(len(bert_tokens))
        
        batch_start2idx.append(start2idx)
        batch_end2idx.append(end2idx)
        
        for e in entity:
            e[0] = start2idx[e[0]]
            e[1] = end2idx[e[1] - 1]
        
        # cut sentence and entity that exceeds max-length
        new_entity, new_bert_tokens = cut_sentence_span(entity, bert_tokens)

        org_sent_len.append(len(new_bert_tokens) - 1)

        new_bert_tokens += [tokenizer.sep_token]
        batch_entities[no] = new_entity

        input_id = tokenizer.convert_tokens_to_ids(new_bert_tokens)
        input_ids.append(input_id) 
    
    max_len = max(map(len, input_ids))
    
    gold_exist, gold_span, gold_iv, exist_mask, span_pool_mask, left_context_mask, right_context_mask = [], [], [], [], [], [], []

    for no, (input_id, entities) in enumerate(zip(input_ids, batch_entities)):
        input_ids[no] += [tokenizer.pad_token_id for _ in range(max_len - len(input_id))]
        emask = torch.triu(torch.ones([len(input_ids[no]), len(input_ids[no])], dtype=torch.long), diagonal=0)
        emask[0] = 0
        emask[:, len(input_id) - 1:] = 0
        exist_matrix = torch.zeros([len(input_ids[no]), len(input_ids[no])], dtype=torch.long)
        for e in entities:
            exist_matrix[e[0], e[1] - 1] = 1
        gold_exist.append(exist_matrix)
        exist_mask.append(emask)

        loc = [[e[0], e[1]] for e in entities]

        pos_span, neg_span = [], []
        pos_iv, neg_iv = [], []
        pos_pool_mask, neg_pool_mask, pos_left_mask, neg_left_mask, pos_right_mask, neg_right_mask = [], [], [], [], [], []

        for start in range(1, org_sent_len[no] + 1):
            if train_mode == "train":
                end_range = min(org_sent_len[no] + 2, start + 11)
            else:
                end_range = org_sent_len[no] + 2
            for end in range(start  + 1, end_range):
                if [start, end] in loc:
                    pos_span.append([start, end, entity_to_idx[entities[loc.index([start, end])][2]], end-start])
                    pos_iv.append(entities[loc.index([start, end])][3])
                    pos_pool_mask.append([1 if start<=i<end else 0 for i in range(len(input_ids[no]))])
                    pos_left_mask.append([1 if 0<i<end else 0 for i in range(len(input_ids[no]))])
                    pos_right_mask.append([1 if start<=i<len(input_id) - 1 else 0 for i in range(len(input_ids[no]))])
                else:
                    neg_span.append([start, end, 0, end-start])
                    neg_iv.append(0.)
                    neg_pool_mask.append([1 if start<=i<end else 0 for i in range(len(input_ids[no]))])
                    neg_left_mask.append([1 if 0<i<end else 0 for i in range(len(input_ids[no]))])
                    neg_right_mask.append([1 if start<=i<len(input_id) - 1 else 0 for i in range(len(input_ids[no]))])

        if train_mode == "train":
            neg_sample = random.sample(list(zip(neg_span, neg_iv, neg_pool_mask, neg_left_mask, neg_right_mask)), min(len(neg_span), sample_count))
            neg_span, neg_iv, neg_pool_mask, neg_left_mask, neg_right_mask = zip(*neg_sample) if neg_sample else ([], [])
        gold_span.append(pos_span + list(neg_span))
        gold_iv.append(pos_iv + list(neg_iv))
        span_pool_mask.append(pos_pool_mask + list(neg_pool_mask))
        left_context_mask.append(pos_left_mask + list(neg_left_mask))
        right_context_mask.append(pos_right_mask + list(neg_right_mask))

    max_span_count = max(map(len, gold_span))

    span_mask = []
    for no, (span, iv) in enumerate(zip(gold_span, gold_iv)):
        gold_span[no].extend([[0, 1, 0, 1] for _ in range(max_span_count - len(span))])
        gold_iv[no].extend([0. for _ in range(max_span_count - len(iv))])
        span_mask.append([1] * len(span) + [0] * (max_span_count - len(span)))
        span_pool_mask[no].extend([[1] + [0] * (len(input_ids[no]) - 1)] * (max_span_count - len(span_pool_mask[no])))
        left_context_mask[no].extend([[1] + [0] * (len(input_ids[no]) - 1)] * (max_span_count - len(left_context_mask[no])))
        right_context_mask[no].extend([[0] * (len(input_ids[no]) - 1) + [1]] * (max_span_count - len(right_context_mask[no])))

    
    return input_ids, gold_exist, gold_span, gold_iv, exist_mask, span_mask, org_sent_len, span_pool_mask, left_context_mask, right_context_mask, batch_start2idx, batch_end2idx 


        


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def save_model(model, optimizer, epoch, scheduler, args):
    checkpoint_save_path = os.path.join(args.checkpoint_path, args.data_mode + '_' + args.model_mode + '_' + str(args.random_seed) + str(args.max_span_len) + '.pth')
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
            'scheduler': scheduler.state_dict()}
    torch.save(state, checkpoint_save_path)	# save the best model

def data_to_device(data_list, device):
    for no in range(len(data_list)):
        data_list[no] = torch.tensor(data_list[no]).to(device)
    return tuple(data_list)



def data_encode(model_mode, dataset, tokenizer, entity_to_idx, train_mode, args, train_dataloader=None):
    data_list = []
    for data in tqdm(dataset):
        input_ids, gold_exist, gold_span, gold_iv, exist_mask, span_mask, org_sent_len, span_pool_mask, left_context_mask, right_context_mask, batch_start2idx, batch_end2idx = sentence_span_encode_v4(data, entity_to_idx, tokenizer, args.sample_count, train_mode)
        gold = 0
        for entities in data['entity']:
            gold += len(entities)
        data_list.append((input_ids, attention_masks(input_ids), gold_exist, gold_span, gold_iv, exist_mask, span_mask, org_sent_len, span_pool_mask, left_context_mask, right_context_mask, gold, batch_start2idx, batch_end2idx))

    return data_list