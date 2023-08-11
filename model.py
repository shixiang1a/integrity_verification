import torch
import torch.nn as nn
from transformers import BertModel
from transformers import BertPreTrainedModel
from layer import SpanAttention
import numpy as np



class InterF(BertPreTrainedModel):
    def __init__(self, config, user_config):
        super(InterF, self).__init__(config)
        self.bert = BertModel(config)
        self.et_dropout = nn.Dropout(user_config.hidden_dropout)
        self.st_dropout = nn.Dropout(user_config.hidden_dropout)
        self.start_layer = torch.nn.Sequential(
            torch.nn.Linear(user_config.hidden_size, user_config.biaffine_output_features),
            torch.nn.ReLU()
        )
        self.end_layer = torch.nn.Sequential(
            torch.nn.Linear(user_config.hidden_size, user_config.biaffine_output_features),
            torch.nn.ReLU()
        )
        self.et_classifier = nn.Linear(user_config.hidden_size + user_config.width_emb_size + 25,  user_config.num_labels)
        # self.et_classifier = nn.Linear(user_config.hidden_size + user_config.width_emb_size,  user_config.num_labels) # without existing
        # self.et_classifier = nn.Linear(user_config.hidden_size + 25,  user_config.num_labels) # without width
        # self.st_classifier = nn.Linear(user_config.hidden_size + user_config.width_emb_size + 25 + 2 * user_config.biaffine_output_features, 1) # emb
        # self.et_classifier = nn.Linear(user_config.hidden_size + 25,  user_config.num_labels)
        # self.et_classifier = nn.Linear(3*config.hidden_size + config.width_emb_size + 25,  config.num_labels)
        # self.et_classifier = nn.Linear(config.hidden_size + config.width_emb_size + 25 + 2 * config.biaffine_output_features,  config.num_labels)
        self.st_classifier = nn.Linear(user_config.hidden_size, 1)
        # self.st_classifier = nn.Linear(user_config.hidden_size + user_config.width_emb_size + 25,  1)
        self.width = nn.Embedding(512, user_config.width_emb_size)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.loss_exist = nn.CrossEntropyLoss(reduction="none")
        self.loss_span = nn.CrossEntropyLoss(reduction="none")
        self.loss_iv = nn.BCELoss(reduction="none")
        self.span_atten_left = SpanAttention(user_config)
        self.span_atten_right = SpanAttention(user_config)
        self.span_atten = SpanAttention(user_config)
        self.num_labels = user_config.num_labels
        self.core = nn.Parameter(torch.Tensor(user_config.biaffine_output_features, user_config.biaffine_output_features, 25))
        self.exist_type = nn.Embedding(2, 25)
        # self.ent_type = nn.Embedding(self.num_labels, 25)
        self.left_gate = nn.Linear(user_config.hidden_size, 1)
        self.right_gate = nn.Linear(user_config.hidden_size, 1)
        # weight initialization
        self.init_weights()


    def get_entity_output(self, start_logits, end_logits, encoder_rep, span_pool_mask, spans, gold_span, span_select, span_width, is_training):
        # m = (span_pool_mask.unsqueeze(-1) == 0).float() * (-1e30)
        # pool_embedding = m + encoder_rep.unsqueeze(1).repeat(1, span_pool_mask.shape[1], 1, 1)
        # pool_embedding = pool_embedding.max(dim=2)[0]

        pool_embedding = []
        pre_exist = []
        for batch_no, (pool_mask, span) in enumerate(zip(span_pool_mask, spans)):
            temp = []
            exist_span = []
            for s in span:
                if s[1].item() == 0:
                    s[1] += 1
                exist_span.append(span_select[batch_no][s[0].item()][s[1].item() - 1])
            pre_exist.append(torch.stack(exist_span))

            for mask in pool_mask:
                m = (mask == 0).float() * (-1e30)
                temp.append((encoder_rep[batch_no] + m.unsqueeze(-1).expand(encoder_rep.shape[1], encoder_rep.shape[2])).max(dim=-2)[0])
            pool_embedding.append(torch.stack(temp))

        pool_embedding = torch.stack(pool_embedding).to(encoder_rep.device)
        pre_exist = torch.stack(pre_exist).to(encoder_rep.device)

        width_embedding = self.width(span_width)
       
        if is_training:
            gold_type = gold_span.clone().detach()
            gold_type[gold_type != 0] = 1
            exist_embedding = self.exist_type(gold_type)
        else:
            exist_embedding = self.exist_type(pre_exist)
        
        # spans_feature = torch.cat([pool_embedding, exist_embedding], dim=-1)
        spans_feature = torch.cat([pool_embedding, width_embedding, exist_embedding], dim=-1)
        # spans_feature = torch.cat([pool_embedding, width_embedding], dim=-1)
        # spans_feature = torch.cat([pool_embedding, width_embedding, exist_embedding, cls_embedding], dim=-1)

        spans_feature = self.et_dropout(spans_feature)
        span_logits = self.et_classifier(spans_feature)
        return span_logits, pool_embedding, exist_embedding, width_embedding

    def get_integrity_output(self, pool_embedding, encoder_rep, attention_mask, left_context_mask, right_context_mask, exist_embedding, ent_select, gold_span, is_training, width_embedding):
        left_context_embedding, atten_left = self.span_atten_left(pool_embedding, encoder_rep, attention_mask, left_context_mask)
        right_context_emebdding, atten_right = self.span_atten_right(pool_embedding, encoder_rep, attention_mask, right_context_mask)
        
        # if is_training:
        #     gold_ent = gold_span[:, :, 2].clone().detach()
        #     ent_embedding = self.ent_type(gold_ent)
        # else:
        #     ent_embedding = self.ent_type(ent_select)
        left_prob = 0
        right_prob = 0

        # sent_feature = torch.cat([left_context_embedding, pool_embedding, right_context_emebdding], dim=-1)
        left_prob = self.sigmoid(self.left_gate(left_context_embedding))
        right_prob = self.sigmoid(self.right_gate(right_context_emebdding))
        sent_feature = left_prob * pool_embedding + right_prob * pool_embedding
        # sent_feature = left_prob * left_context_embedding + right_prob * right_context_emebdding + pool_embedding
        # sent_feature, _ = self.span_atten(pool_embedding, encoder_rep, attention_mask)
        # sent_feature = pool_embedding
        # sent_feature = torch.cat([pool_embedding, width_embedding, exist_embedding], dim=-1)

        sent_feature = self.st_dropout(sent_feature)
        sentence_logits = self.st_classifier(sent_feature)

        sentence_prob = self.sigmoid(sentence_logits)
        return sentence_prob, atten_left, atten_right, left_prob, right_prob

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, span_pool_mask=None, gold_exist=None, gold_span=None, gold_iv=None, exist_mask=None, span_mask=None, left_context_mask=None, right_context_mask=None, span_width=None, spans=None, is_training=True):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_rep  = outputs[0]
        start_logits = self.start_layer(encoder_rep) 
        end_logits = self.end_layer(encoder_rep)
        core = self.core.unsqueeze(0).repeat(encoder_rep.shape[0], 1, 1, 1)
        e_type = self.exist_type(torch.tensor([0, 1]).to(encoder_rep.device)).unsqueeze(0).repeat(encoder_rep.shape[0], 1, 1)

        tucker = torch.einsum('bij, bjmo, bkm, blo->bikl', start_logits, core, end_logits, e_type)

        span_exist = self.softmax(tucker)

        span_select = span_exist.clone().detach()
        span_select = torch.max(span_select, dim=-1)[1]


        # entity recgonition layer
        span_logits, pool_embedding, exist_embedding, width_embedding = self.get_entity_output(start_logits, end_logits, encoder_rep, span_pool_mask, spans, gold_span, span_select, span_width, is_training)
        span_prob = self.softmax(span_logits)

        ent_select = span_prob.clone().detach()
        ent_select = torch.max(ent_select, dim=-1)[1]
        # ent_select[ent_select != 0] = 1

        # integrity layer
        sentence_prob, atten_left, atten_right, left_prob, right_prob = self.get_integrity_output(pool_embedding, encoder_rep, attention_mask, left_context_mask, right_context_mask, exist_embedding, ent_select, gold_span, is_training, width_embedding)
        

        # calculate loss
        loss_exist = torch.sum(self.loss_exist(tucker.contiguous().view(-1, 2), gold_exist.view(-1)) * exist_mask.view(-1)) / torch.sum(exist_mask)
        loss_span = torch.sum(self.loss_span(span_logits.contiguous().view(-1, self.num_labels), gold_span.view(-1)) * span_mask.view(-1)) / torch.sum(span_mask)
        loss_iv = torch.sum(self.loss_iv(sentence_prob.contiguous().view(-1), gold_iv.view(-1)) * span_mask.view(-1)) / torch.sum(span_mask)

        loss_all = loss_span + loss_iv + loss_exist
        # loss_all = loss_span + loss_iv

        if is_training:
            return loss_all, span_prob, sentence_prob, loss_span, loss_iv, loss_exist
        else:
            return loss_all, span_exist, span_prob, sentence_prob, loss_span, loss_iv, loss_exist, atten_left, atten_right, left_prob, right_prob
        


class InterF_ent(BertPreTrainedModel):
    def __init__(self, config, user_config):
        super(InterF_ent, self).__init__(config)
        self.bert = BertModel(config)
        self.et_dropout = nn.Dropout(user_config.hidden_dropout)
        self.st_dropout = nn.Dropout(user_config.hidden_dropout)
        self.start_layer = torch.nn.Sequential(
            torch.nn.Linear(user_config.hidden_size, user_config.biaffine_output_features),
            torch.nn.ReLU()
        )
        self.end_layer = torch.nn.Sequential(
            torch.nn.Linear(user_config.hidden_size, user_config.biaffine_output_features),
            torch.nn.ReLU()
        )
        self.et_classifier = nn.Linear(user_config.hidden_size + user_config.width_emb_size + 25,  user_config.num_labels)
        self.width = nn.Embedding(512, user_config.width_emb_size)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()
        self.loss_exist = nn.CrossEntropyLoss(reduction="none")
        self.loss_span = nn.CrossEntropyLoss(reduction="none")
        self.num_labels = user_config.num_labels
        self.core = nn.Parameter(torch.Tensor(user_config.biaffine_output_features, user_config.biaffine_output_features, 25))
        self.exist_type = nn.Embedding(2, 25)
    
    
    def get_entity_output(self, start_logits, end_logits, encoder_rep, span_pool_mask, spans, gold_span, span_select, span_width, is_training):
        # m = (span_pool_mask.unsqueeze(-1) == 0).float() * (-1e30)
        # pool_embedding = m + encoder_rep.unsqueeze(1).repeat(1, span_pool_mask.shape[1], 1, 1)
        # pool_embedding = pool_embedding.max(dim=2)[0]

        pool_embedding = []
        pre_exist = []
        for batch_no, (pool_mask, span) in enumerate(zip(span_pool_mask, spans)):
            temp = []
            exist_span = []
            for s in span:
                if s[1].item() == 0:
                    s[1] += 1
                exist_span.append(span_select[batch_no][s[0].item()][s[1].item() - 1])
            pre_exist.append(torch.stack(exist_span))

            for mask in pool_mask:
                m = (mask == 0).float() * (-1e30)
                temp.append((encoder_rep[batch_no] + m.unsqueeze(-1).expand(encoder_rep.shape[1], encoder_rep.shape[2])).max(dim=-2)[0])
            pool_embedding.append(torch.stack(temp))

        pool_embedding = torch.stack(pool_embedding).to(encoder_rep.device)
        pre_exist = torch.stack(pre_exist).to(encoder_rep.device)

        width_embedding = self.width(span_width)
        
        # start_logtis [batch_size, sent_length, biaffine_hiiden]
        # gold_span [batch_size, span_lenth]
        # span_start_embedding = self.batched_index_select(start_logits, 1, spans[:, :, 0])

        # span_end_embedding = self.batched_index_select(end_logits, 1, spans[:, :, 1] - 1)
       
        if is_training:
            gold_type = gold_span.clone().detach()
            gold_type[gold_type != 0] = 1
            exist_embedding = self.exist_type(gold_type)
        else:
            exist_embedding = self.exist_type(pre_exist)
        
        spans_feature = torch.cat([pool_embedding, width_embedding, exist_embedding], dim=-1)

        spans_feature = self.et_dropout(spans_feature)
        span_logits = self.et_classifier(spans_feature)
        return span_logits

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, span_pool_mask=None, gold_exist=None, gold_span=None, exist_mask=None, span_mask=None, span_width=None, spans=None, is_training=True):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        encoder_rep  = outputs[0]
        start_logits = self.start_layer(encoder_rep) 
        end_logits = self.end_layer(encoder_rep)
        core = self.core.unsqueeze(0).repeat(encoder_rep.shape[0], 1, 1, 1)
        e_type = self.exist_type(torch.tensor([0, 1]).to(encoder_rep.device)).unsqueeze(0).repeat(encoder_rep.shape[0], 1, 1)

        tucker = torch.einsum('bij, bjmo, bkm, blo->bikl', start_logits, core, end_logits, e_type)

        span_exist = self.softmax(tucker)

        span_select = span_exist.clone().detach()
        span_select = torch.max(span_select, dim=-1)[1]


        # entity recgonition layer
        span_logits = self.get_entity_output(start_logits, end_logits, encoder_rep, span_pool_mask, spans, gold_span, span_select, span_width, is_training)
        span_prob = self.softmax(span_logits)

        ent_select = span_prob.clone().detach()
        ent_select = torch.max(ent_select, dim=-1)[1]

        # calculate loss
        loss_exist = torch.sum(self.loss_exist(tucker.contiguous().view(-1, 2), gold_exist.view(-1)) * exist_mask.view(-1)) / torch.sum(exist_mask)
        loss_span = torch.sum(self.loss_span(span_logits.contiguous().view(-1, self.num_labels), gold_span.view(-1)) * span_mask.view(-1)) / torch.sum(span_mask)

        loss_all = loss_span + loss_exist

        if is_training:
            return loss_all, span_prob, loss_span, loss_exist
        else:
            return loss_all, span_exist, span_prob, loss_span, loss_exist