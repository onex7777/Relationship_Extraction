#! -*- coding:utf-8 -*-
import numpy as np
import json
from random import choice

BERT_MAX_LEN = 256
RANDOM_SEED = 2021


def find_head_idx(source, target):
    target_len = len(target)
    for i in range(len(source)):
        if source[i: i + target_len] == target:
            return i
    return -1


def to_tuple(sent):
    triple_list = []
    for triple in sent['triple_list']:
        triple_list.append(tuple(triple))
    sent['triple_list'] = triple_list


def seq_padding(batch, padding=0, ):
    length_batch = [len(seq) for seq in batch]
    max_length = max(length_batch)
    x = np.array([
        np.concatenate([seq, [padding] * (max_length - len(seq))]) if len(seq) < max_length else seq for seq in batch
    ])
    return x


def load_data(train_path, dev_path, test_path, rel_dict_path):
    train = open(train_path, encoding='utf8')
    dev = open(dev_path, encoding='utf8')
    test = open(test_path, encoding='utf8')
    rel = open(rel_dict_path, encoding='utf8')
    train_data = json.load(train)
    dev_data = json.load(dev)
    test_data = json.load(test)
    id2rel, rel2id = json.load(rel)

    id2rel = {int(i): j for i, j in id2rel.items()}
    num_rels = len(id2rel)

    random_order = list(range(len(train_data)))
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(random_order)
    train_data = [train_data[i] for i in random_order]

    for sent in train_data:
        to_tuple(sent)
    for sent in dev_data:
        to_tuple(sent)
    for sent in test_data:
        to_tuple(sent)

    print("train_data len:", len(train_data))
    print("dev_data len:", len(dev_data))
    print("test_data len:", len(test_data))
    train.close()
    dev.close()
    test.close()
    rel.close()
    return train_data, dev_data, test_data, id2rel, rel2id, num_rels


class data_generator:
    def __init__(self, data, tokenizer, rel2id, num_rels, maxlen, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.rel2id = rel2id
        self.num_rels = num_rels

    def __len__(self):
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
        return self.steps

    def __iter__(self):
        # dataset = []
        while True:
            idxs = list(range(len(self.data)))
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(idxs)
            tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch = [], [], [], [], [], [], [], []
            for idx in idxs:
                line = self.data[idx]
                text = ' '.join(line['text'].split()[:self.maxlen])
                tokens = self.tokenizer.tokenize(text)
                # print(text)
                # print(tokens)
                if len(tokens) > BERT_MAX_LEN:
                    tokens = tokens[:BERT_MAX_LEN]
                text_len = len(tokens)

                s2ro_map = {}  # 得到{sub：obj和关系}的坐标的首尾坐标{(29, 31): [(23, 25, 11)], (23, 25): [(19, 20, 29)]......}
                for triple in line['triple_list']:
                    triple = (
                        self.tokenizer.tokenize(triple[0])[1:-1], triple[1], self.tokenizer.tokenize(triple[2])[1:-1])
                    sub_head_idx = find_head_idx(tokens, triple[0])
                    obj_head_idx = find_head_idx(tokens, triple[2])
                    if sub_head_idx != -1 and obj_head_idx != -1:
                        sub = (sub_head_idx, sub_head_idx + len(triple[0]) - 1)
                        if sub not in s2ro_map:
                            s2ro_map[sub] = []
                        s2ro_map[sub].append((obj_head_idx,
                                              obj_head_idx + len(triple[2]) - 1,
                                              self.rel2id[triple[1]]))

                if s2ro_map:

                    token_ids, segment_ids = self.tokenizer.encode(text)
                    if len(token_ids) > text_len:
                        token_ids = token_ids[:text_len]
                        segment_ids = segment_ids[:text_len]
                    tokens_batch.append(token_ids)  # word embedding
                    segments_batch.append(segment_ids)  # 句子embedding
                    sub_heads, sub_tails = np.zeros(text_len), np.zeros(text_len)
                    for s in s2ro_map.keys():
                        sub_heads[s[0]] = 1
                        sub_tails[s[1]] = 1
                    sub_head, sub_tail = choice(list(s2ro_map.keys()))
                    obj_heads, obj_tails = np.zeros((text_len, self.num_rels)), np.zeros((text_len, self.num_rels))
                    for ro in s2ro_map.get((sub_head, sub_tail), []):  # ro = (25, 27, 31)为obj的首末和关系id
                        obj_heads[ro[0]][ro[2]] = 1
                        obj_tails[ro[1]][ro[2]] = 1  # 得到obj实体的tail和关系id
                    sub_heads_batch.append(sub_heads)
                    sub_tails_batch.append(sub_tails)
                    sub_head_batch.append(
                        [sub_head])  # 单个sub的头的坐标[[14], [ 2], [10], [19], [17], [ 1], [16], [ 5]]目前只是list
                    sub_tail_batch.append(
                        [sub_tail])  # 单个sub的尾的坐标[[18], [19], [12], [21], [18], [ 4], [21], [20]]目前只是list
                    obj_heads_batch.append(obj_heads)
                    obj_tails_batch.append(obj_tails)
                    if len(tokens_batch) == self.batch_size or idx == idxs[-1]:
                        tokens_batch = seq_padding(tokens_batch)
                        segments_batch = seq_padding(segments_batch)  # [batch,len]
                        sub_heads_batch = seq_padding(sub_heads_batch)
                        sub_tails_batch = seq_padding(sub_tails_batch)  # [batch,len]
                        obj_heads_batch = seq_padding(obj_heads_batch, padding=np.zeros(self.num_rels))  # [batch,len,num_rels]
                        obj_tails_batch = seq_padding(obj_tails_batch, padding=np.zeros(self.num_rels))
                        sub_head_batch, sub_tail_batch = np.array(sub_head_batch), np.array(sub_tail_batch)  # [batch,1]
                        yield [tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch,
                               sub_tail_batch, obj_heads_batch, obj_tails_batch], None
                        tokens_batch, segments_batch, sub_heads_batch, sub_tails_batch, sub_head_batch, sub_tail_batch, obj_heads_batch, obj_tails_batch, = [], [], [], [], [], [], [], []


if __name__ == "__main__":
    from utils import get_tokenizer
    # pre-trained bert save_model config
    bert_model = 'weight'
    bert_config_path = bert_model + '/config.json'
    bert_vocab_path = bert_model + '/vocab.txt'
    bert_checkpoint_path = bert_model + '/bert_model.ckpt'

    dataset_dir = 'DuIE_2_0/triples'
    train_path = dataset_dir + '/train_triples.json'
    dev_path = dataset_dir + '/dev_triples.json'
    test_path = dataset_dir + '/test_triples.json'
    rel_dict_path = dataset_dir + '/rel2id.json'
    save_weights_path = './checkpoint/'
    log_dir = "./log_dir/"
    BATCH_SIZE = 16
    MAX_LEN = 256
    tokenizer = get_tokenizer(bert_vocab_path)
    train_data, dev_data, test_data, id2rel, rel2id, num_rels = load_data(train_path, dev_path, test_path,
                                                                          rel_dict_path)
    data_manager = data_generator(train_data, tokenizer, rel2id, num_rels, MAX_LEN, BATCH_SIZE)
    data_manager.__iter__()