
import os
import sys
import json
from tqdm import tqdm
import codecs
import numpy as np
import pandas as pd 
from bert4keras.tokenizers import Tokenizer
raw_data_dir = os.getcwd()
training_data_dir = os.getcwd()
def get_tokenizer(vocab_path):
    tokenizer = Tokenizer(vocab_path, do_lower_case=True)
    return tokenizer
tokenizer = get_tokenizer(r'D:\Python\Tensorflow\natural_language_processing\Relationship_Extraction\weight\vocab.txt')
RANDOM_SEED = 2019

rel_set = set()

text_len = []

train_data = []
i= 0
with open(raw_data_dir + '/DuIE_2_0/train.json', encoding='utf8') as f:
    for l in tqdm(f.readlines()):
        a = json.loads(l)
        if i == 0:
            print(json.dumps(a, sort_keys=True, indent=4, separators=(', ', ': '),ensure_ascii=False))
        if len(tokenizer.tokenize(' '.join(json.loads(l)["text"].strip().split(" ")).lower())) > 78:
            continue
        if not a['spo_list']:
            continue
        triple_list = []
        for spo in a['spo_list']:
            s = spo['subject']
            p = spo['predicate']
            o_dict = spo['object']
            for k in o_dict.keys():
                triple_list.append((s, p+'_'+k, o_dict[k]))
                rel_set.add(p+'_'+k)

        line = {
                'text': a['text'],
                'triple_list': triple_list
               }
        if i == 0:
            print(json.dumps(line, sort_keys=True, indent=4, separators=(', ', ': '),ensure_ascii=False))
        train_data.append(line)
        text_len.append((len(tokenizer.tokenize(' '.join(json.loads(l)["text"].strip().split(" ")).lower()))))
        i += 1

df = pd.DataFrame({"text_len":text_len})
print("训练集文本长度统计：\n")
print(df["text_len"].describe())

id2rel = {i:j for i,j in enumerate(sorted(rel_set))}
rel2id = {j:i for i,j in id2rel.items()}

with codecs.open(training_data_dir+'/DuIE_2_0/triples/rel2id.json', 'w', encoding='utf-8') as f:
    json.dump([id2rel, rel2id], f, indent=4, ensure_ascii=False)

with codecs.open(training_data_dir+'/DuIE_2_0/triples/train_triples.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)


dev_data = []

with open(raw_data_dir + '/DuIE_2_0/dev.json', encoding='utf8') as f:
    for l in tqdm(f.readlines()):
        a = json.loads(l)
        if not a['spo_list']:
            continue
        if len(tokenizer.tokenize(' '.join(json.loads(l)["text"].strip().split(" ")))) > 78:
            continue
        triple_list = []
        for spo in a['spo_list']:
            s = spo['subject']
            p = spo['predicate']
            o_dict = spo['object']
            for k in o_dict.keys():
                triple_list.append((s, p+'_'+k,o_dict[k]))
                rel_set.add(p+'_'+k)

        line = {
                'text': a['text'],
                'triple_list': triple_list
               }
        dev_data.append(line)


dev_len = len(dev_data)
random_order = list(range(dev_len))
np.random.seed(RANDOM_SEED)
np.random.shuffle(random_order)

test_data = [dev_data[i] for i in random_order[:int(0.5 * dev_len)]]
dev_data = [dev_data[i] for i in random_order[int(0.5 * dev_len):]]

with codecs.open(training_data_dir+'/DuIE_2_0/triples/dev_triples.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)

with codecs.open(training_data_dir+'/DuIE_2_0/triples/test_triples.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)