import json
import os
from torch.utils.data import Dataset, DataLoader


class NERdataLoader(Dataset):
    def __init__(self, file_path):
        super(NERdataLoader, self).__init__()
        self.file_path = file_path
        self.file = open(self.file_path, 'r', encoding='utf-8')
        self.res = self.file.readlines()
        
    def __getitem__(self, idx):
        dic = self.res[idx]
        jdata = json.loads(dic)
        sentence = jdata['sentence']
        entities = jdata['ner']
        integrity = jdata['integrity']
        if integrity:
            label = 1.
        else:
            label = 0.
        for e in entities:
            e[1] = e[1] + 1
            if e[3]:
                e[3] == 1.
            else:
                e[3] == 0.
        line = {'sentence': sentence, 'entity': entities, 'id': idx, 'integrity': label}
        return line
                 
    def __len__(self):
        return len(self.res)
    
    def get_entity_to_idx(self):
        entity_type = []
        for dic in self.res:
            jdata = json.loads(dic)
            entities = jdata['ner']
            for e in entities:
                if e[2] not in entity_type:
                    entity_type.append(e[2])
        entity_to_idx = {i: no + 1 for no, i in enumerate(entity_type)}
        entity_to_idx['None'] = 0
        return entity_to_idx


def collate_batch(batch):
    sentence = []
    entities = []
    idx = []
    integrity = []
    for sample in batch:
        sentence.append(sample['sentence'])
        entities.append(sample['entity'])
        idx.append(sample['id'])
        integrity.append(sample['integrity'])
    return {'sentence': sentence, 'entity': entities, 'id': idx, 'integrity': integrity}


def data_split(file_path):
    file = open(file_path, 'r', encoding='utf-8')
    dataset = file.readlines()
    if os.path.exists(file_path.replace('.json', '_split.json')):
        return file_path.replace('.json', '_split.json')
    new_file = open(file_path.replace('.json', '_split.json'), 'a', encoding='utf-8')
    for dic in dataset:
        jdata = json.loads(dic)
        entities = jdata['ner']
        sentence = jdata['sentence']
        sub_sentence = []
        sub_entities = []
        sub_integrity = True
        for no, token in enumerate(sentence):
            if token == '.' or no == len(sentence) - 1:
                sub_sentence.append(token)
                for e in entities:
                    if e[1] < len(sub_sentence) and e[0] >= 0:
                        sub_entities.append([e[0], e[1], e[2], e[3]])
                        if not e[3]:
                            sub_integrity = False
                for e in entities:
                    e[0] -= len(sub_sentence)
                    e[1] -= len(sub_sentence)
                new_file.write(json.dumps({"integrity": sub_integrity, "sentence": sub_sentence, "ner": sub_entities}))
                new_file.write('\n')
                sub_sentence = []
                sub_entities = []
                sub_integrity = True
            else:
                sub_sentence.append(token)
    return file_path.replace('.json', '_split.json')
                


def train_data(batch_size, data_dir, data_mode, args):
    if args.split_sent:
        train_path = data_split(os.path.join(data_dir, data_mode + '/', 'train.json'))
    else:
        train_path = os.path.join(data_dir, data_mode + '/', 'train.json')

    nerdataLoader = NERdataLoader(train_path)
    # print(nerdataLoader.__len__())
    entity_to_idx = nerdataLoader.get_entity_to_idx()
    with open(data_dir + 'config/' + data_mode + '.json', 'w') as f:
        f.write(json.dumps(entity_to_idx, ensure_ascii=False))
    train_dataset = DataLoader(dataset=nerdataLoader, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=False, collate_fn=collate_batch)
    return train_dataset, entity_to_idx, nerdataLoader


def eval_data(test_mode, batch_size, data_dir, data_mode, args):
    
    if test_mode == 'dev':
        if args.split_sent:
            dev_path = data_split(os.path.join(data_dir, data_mode + '/', 'dev.json'))
        else:
            dev_path = os.path.join(data_dir, data_mode + '/', 'dev.json')
        nerdataLoader = NERdataLoader(dev_path)
    elif test_mode == 'test':
        if args.split_sent:
            test_path = data_split(os.path.join(data_dir, data_mode + '/', 'test.json'))
        else:
            test_path = os.path.join(data_dir, data_mode + '/', 'test.json')
        nerdataLoader = NERdataLoader(test_path)
    else:
        nerdataLoader = NERdataLoader(test_mode)  
    dataset = DataLoader(dataset=nerdataLoader, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False, collate_fn=collate_batch)
    return dataset


class PredictdataLoader(Dataset):
    def __init__(self, file_path):
        super(PredictdataLoader, self).__init__()
        self.file_path = file_path
        self.file = open(self.file_path, encoding='utf-8')
        self.res = self.file.readlines()
        
    def __getitem__(self, idx):
        dic = self.res[idx]
        jdata = json.loads(dic)
        sentence = jdata['sentence']
        line = {'sentence': sentence, 'id': idx}
        return line
                 
    def __len__(self):
        return len(self.res)


def predict_collate_batch(batch):
    sentence = []
    idx = []
    for sample in batch:
        sentence.append(sample['sentence'])
        idx.append(sample['id'])
    return {'sentence': sentence, 'id': idx}

# For Spancls
class PredictdataLoader_v2(Dataset):
    def __init__(self, file_path):
        super(PredictdataLoader_v2, self).__init__()
        self.file_path = file_path
        self.file = open(self.file_path, encoding='utf-8')
        self.res = self.file.readlines()
        
    def __getitem__(self, idx):
        dic = self.res[idx]
        jdata = json.loads(dic)
        sentence = jdata['sentence']
        entities = jdata['ner']
        for e in entities:
            e[1] = e[1] + 1
        line = {'sentence': sentence, 'entity': entities, 'id': idx}
        return line
             
    def __len__(self):
        return len(self.res)


def predict_collate_batch_v2(batch):
    sentence = []
    idx = []
    entities = []
    for sample in batch:
        sentence.append(sample['sentence'])
        idx.append(sample['id'])
        entities.append(sample['entity'])
    return {'sentence': sentence, 'entity': entities, 'id': idx}



def predict_data(data_path, batch_size, args):
    if args.model_mode == 'Span-cls':
        predataloader_v2 = PredictdataLoader_v2(data_path)
        dataset = DataLoader(dataset=predataloader_v2, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False, collate_fn=predict_collate_batch_v2)
    else:
        predataloader = PredictdataLoader(data_path)
        dataset = DataLoader(dataset=predataloader, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=False, collate_fn=predict_collate_batch)
    return dataset


