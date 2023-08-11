import random

import torch

import util
import os
import numpy as np



def create_train_sample(doc, neg_entity_count: int, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # positive entities
    pos_entity_spans, pos_entity_types, pos_sentence_types, pos_entity_masks, pos_entity_sizes, pos_left_masks, pos_right_masks = [], [], [], [], [], [], []
    for e in doc.entities:
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_sentence_types.append(e.sentence_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_sizes.append(len(e.tokens))
        pos_left_masks.append([1 if 0<k<e.span_end else 0 for k in range(context_size)])
        pos_right_masks.append([1 if e.span_start<=k<context_size - 1 else 0 for k in range(context_size)])
    # negative entities
    neg_entity_spans, neg_entity_sizes, neg_left_masks, neg_right_masks = [], [], [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)
                neg_left_masks.append([1 if 0<k<span[1] else 0 for k in range(context_size)])
                neg_right_masks.append([1 if span[0]<=k<context_size - 1 else 0 for k in range(context_size)])

    # sample negative entities
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes, neg_left_masks, neg_right_masks)),
                                       min(len(neg_entity_spans), neg_entity_count))
    neg_entity_spans, neg_entity_sizes, neg_left_masks, neg_right_masks = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)
    neg_sentence_types = [0] * len(neg_entity_spans)

    # merge
    entity_types = pos_entity_types + neg_entity_types
    entity_masks = pos_entity_masks + neg_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes)
    sentence_types = pos_sentence_types + neg_sentence_types
    left_masks = pos_left_masks + list(neg_left_masks)
    right_masks = pos_right_masks + list(neg_right_masks)
    entity_spans = list(pos_entity_spans) + list(neg_entity_spans)

    assert len(entity_masks) == len(entity_sizes) == len(entity_types) == len(sentence_types) == len(right_masks) == len(left_masks)

    # create tensors
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    # also create samples_masks:
    # tensors to mask entity/relation samples of batch
    # since samples are stacked into batches, "padding" entities/relations possibly must be created
    # these are later masked during loss computation
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        sentence_types = torch.tensor(sentence_types, dtype=torch.float)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
        left_masks = torch.tensor(left_masks)
        right_masks = torch.tensor(right_masks)
        entity_spans = torch.tensor(entity_spans)
    else:
        # corner case handling (no pos/neg entities)
        entity_types = torch.zeros([1], dtype=torch.long)
        sentence_types = torch.zeros([1], dtype=torch.float)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
        left_masks = torch.zeros([1, context_size], dtype=torch.bool)
        right_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_types=entity_types, sentence_types=sentence_types,
                entity_sample_masks=entity_sample_masks, left_masks=left_masks, right_masks=right_masks, entity_spans=entity_spans)


def create_eval_sample(doc, max_span_size: int):
    encodings = doc.encoding
    token_count = len(doc.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []
    left_masks = []
    right_masks = []
    sentence_types = []
    entity_types = []
    pos_entity_spans = []

    for e in doc.entities:
        pos_entity_spans.append(e.span)


    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(size)
            left_masks.append([1 if 0<k<span[1] else 0 for k in range(context_size)])
            right_masks.append([1 if span[0]<=k<context_size - 1 else 0 for k in range(context_size)])
            if span in pos_entity_spans:
                sentence_types.append(doc.entities[pos_entity_spans.index(span)].sentence_type.index)
                entity_types.append(doc.entities[pos_entity_spans.index(span)].entity_type.index)
            else:
                sentence_types.append(0.)
                entity_types.append(0)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)
        left_masks = torch.tensor(left_masks)
        right_masks = torch.tensor(right_masks)
        sentence_types = torch.tensor(sentence_types)
        entity_types = torch.tensor(entity_types, dtype=torch.long)


        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
        left_masks = torch.zeros([1, context_size], dtype=torch.bool)
        right_masks = torch.zeros([1, context_size], dtype=torch.bool)
        sentence_types = torch.zeros([1], dtype=torch.float)
        entity_types = torch.zeros([1], dtype=torch.long)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks, left_masks=left_masks, right_masks=right_masks,
                sentence_types=sentence_types, entity_types=entity_types)


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
