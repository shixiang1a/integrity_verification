import json
from abc import abstractmethod, ABC
from collections import OrderedDict
from logging import Logger
from typing import List
from tqdm import tqdm
from transformers import BertTokenizer

import util
from entities import Dataset, EntityType, Entity, Document
from opt import spacy


# 数据读取父类
class BaseInputReader(ABC):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 max_span_size: int = None, logger: Logger = None, **kwargs):
        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity types
        
        # 实例类型字典
        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._sentence_types = OrderedDict()
        self._idx2sentence_type = OrderedDict()

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # integrity type
        for i, (key, v) in enumerate(types['integrity'].items()):
            sentence_type = EntityType(key, i, v['short'], v['verbose'])
            self._sentence_types[key] = sentence_type
            self._idx2sentence_type[i] = sentence_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type

        self._neg_entity_count = neg_entity_count
        self._max_span_size = max_span_size

        self._datasets = dict()

        self._tokenizer = tokenizer
        self._logger = logger

        self._vocabulary_size = tokenizer.vocab_size

    @abstractmethod
    def read(self, dataset_path, dataset_label):
        pass

    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity
    
    def get_sentence_type(self, idx) -> EntityType:
        sentence = self._idx2sentence_type[idx]
        return sentence

    def _log(self, text):
        if self._logger is not None:
            self._logger.info(text)

    @property
    def datasets(self):
        return self._datasets

    @property
    def entity_types(self):
        return self._entity_types
    
    @property
    def sentence_types(self):
        return self._sentence_types

    @property
    def entity_type_count(self):
        return len(self._entity_types)
    
    @property
    def sentence_type_count(self):
        return len(self._sentence_types)

    @property
    def vocabulary_size(self):
        return self._vocabulary_size

    def __str__(self):
        string = ""
        for dataset in self._datasets.values():
            string += "Dataset: %s\n" % dataset
            string += str(dataset)

        return string

    def __repr__(self):
        return self.__str__()

# 数据读取子类 继承 BaseInputReader
class JsonInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, neg_entity_count: int = None,
                 max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, neg_entity_count, max_span_size, logger)

    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label, self._entity_types, self._sentence_types, self._neg_entity_count, self._max_span_size)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        documents = []
        res = open(dataset_path).readlines()
        for line in res:
            documents.append(json.loads(line))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)
    
    # 每个token、entity和relation都单独创建类实例
    def _parse_document(self, doc, dataset) -> Document:
        jtokens = doc['sentence']
        jentities = doc['ner']

        # parse tokens
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)

        # parse entity mentions
        entities = self._parse_entities(jentities, doc_tokens, dataset)

        # create document
        document = dataset.create_document(doc_tokens, entities, doc_encoding)
        return document
        # if len(doc_encoding) <= 512:
        #     document = dataset.create_document(doc_tokens, entities, doc_encoding)
        #     return document
        # else:
        #     return 0

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            entity_type = self._entity_types[jentity[2]]
            sentence_type = self._sentence_types[str(jentity[3])]
            start, end = jentity[0], jentity[1] + 1

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, sentence_type, tokens, phrase)
            entities.append(entity)

        return entities



class JsonPredictionInputReader(BaseInputReader):
    def __init__(self, types_path: str, tokenizer: BertTokenizer, spacy_model: str = None,
                 max_span_size: int = None, logger: Logger = None):
        super().__init__(types_path, tokenizer, max_span_size=max_span_size, logger=logger)
        self._spacy_model = spacy_model

        self._nlp = spacy.load(spacy_model) if spacy is not None and spacy_model is not None else None

    def read(self, dataset_path, dataset_label):
        dataset = Dataset(dataset_label, self._entity_types, self._neg_entity_count, self._max_span_size)
        self._parse_dataset(dataset_path, dataset)
        self._datasets[dataset_label] = dataset
        return dataset

    def _parse_dataset(self, dataset_path, dataset):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % dataset.label):
            self._parse_document(document, dataset)

    def _parse_document(self, document, dataset) -> Document:
        if type(document) == list:
            jtokens = document
        elif type(document) == dict:
            jtokens = document['tokens']
        else:
            jtokens = [t.text for t in self._nlp(document)]

        # parse tokens
        doc_tokens, doc_encoding = _parse_tokens(jtokens, dataset, self._tokenizer)

        # create document
        document = dataset.create_document(doc_tokens, [], doc_encoding)

        return document


# 句子encode BPE
def _parse_tokens(jtokens, dataset, tokenizer):
    doc_tokens = []

    # full document encoding including special tokens ([CLS] and [SEP]) and byte-pair encodings of original tokens
    doc_encoding = [tokenizer.convert_tokens_to_ids('[CLS]')]

    # parse tokens
    for i, token_phrase in enumerate(jtokens):
        token_encoding = tokenizer.encode(token_phrase, add_special_tokens=False)
        if not token_encoding:
            token_encoding = [tokenizer.convert_tokens_to_ids('[UNK]')]
        span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

        token = dataset.create_token(i, span_start, span_end, token_phrase)

        doc_tokens.append(token)
        doc_encoding += token_encoding

    doc_encoding += [tokenizer.convert_tokens_to_ids('[SEP]')]

    return doc_tokens, doc_encoding
