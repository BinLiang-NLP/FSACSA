# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from collections import Counter
from copy import deepcopy
from glob import glob
import math
import json
from random import sample
import pandas as pd
import argparse
TOTAL_ASPECTS = ["food","menu","ambience","miscellaneous","place","service","price","staff",
                 "FOOD#QUALITY","RESTAURANT#PRICES","AMBIENCE#GENERAL","RESTAURANT#MISCELLANEOUS",
                 "FOOD#PRICES","DRINKS#QUALITY","DRINKS#PRICES","RESTAURANT#GENERAL","FOOD#STYLE_OPTIONS","SERVICE#GENERAL"]


def build_tokenizer_old(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer

def split_punc(text):
    processed_text = ""
    for c in text:
        if c in [',','.','!','?','/','#','@','(',')','{','}']:
            processed_text += ' '
        processed_text += c
    return processed_text

def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines)):
                line = lines[i].strip()
                line_data = line.split("\t")
                assert len(line_data)==2
                # text_raw = line_data[0]
                text_raw = split_punc(line_data[0])
                aspect = fname.split("/")[-1].lower()
                aspect.replace("_"," ")
                # assert aspect in TOTAL_ASPECTS,"{} is new".format(aspect)
                text_raw = text_raw + " " + aspect.replace("#"," ")+ " "
                text += text_raw
        tokenizer = Tokenizer(max_seq_len)
        # print(text)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else '../glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x



class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {'[mask]':1}
        self.idx2word = {1:'[mask]'}
        self.idx = 2

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post',max_seq_len=None):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        return pad_and_truncate(sequence, max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        # self.tokenizer.add_special_tokens()
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post',max_seq_len=None):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        return pad_and_truncate(sequence, max_seq_len, padding=padding, truncating=truncating)



class FewshotACSADataset():
    def __init__(self, data_dir, tasks, tokenizer, opt):
        self.tasks = tasks
        self.all_aspects = self.tasks
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.opt = opt
        self.polarities = opt.polarities
        self.get_all_aspect_indices()
        self._build_seed_dict()
        self._get_all_data()

    def get_all_aspect_indices(self,):
        self.all_aspect_indices = []
        for aspect in self.all_aspects:
            aspect  = aspect.replace("#",' ')
            aspect = aspect.replace("_",' ')
            aspect_indices = self.tokenizer.text_to_sequence(aspect,max_seq_len=4)
            assert not all(aspect_indices==0), "aspect error"
            self.all_aspect_indices.append(aspect_indices)
        assert len(self.all_aspect_indices)==len(self.tasks)





    def _build_seed_dict(self):
        self.seed_dict = {}
        for file in self.opt.dataset_files['seed_files']:
            seed_df = pd.read_csv(file)
            for idx, row in seed_df.iterrows():
                aspect = row[0].lower()
                seed_words = []
                for i in range(1,6):
                    seed_words += eval(row[i])
                self.seed_dict[aspect] = seed_words



    def get_masked_text(self,text,aspect):
        candidate_words = []
        for aspect_word in aspect.split(" "):
            seed_words = self.seed_dict.get(aspect_word)
            if seed_words:
                candidate_words += seed_words
        orgin_words = text.split(' ')
        masked_words = [ wd for wd in orgin_words]
        keep_seed_words = [ '[MASK]' for wd in orgin_words]
        masked_num = 0
        for idx, word in enumerate(orgin_words):
            if word in candidate_words:
                masked_words[idx] = '[MASK]'
                keep_seed_words[idx] = word
                masked_num +=1
        masked_text = " ".join(masked_words)
        seed_text  = " ".join(keep_seed_words)
        return masked_text, seed_text


    def _get_all_data(self):
        self.all_data = {} ## task_name:data_list
        self.polarity2label = { polarity:idx for idx, polarity in enumerate(self.opt.polarities)}
        self.label2polarity = { idx:polarity for idx, polarity in enumerate(self.opt.polarities)}
        out_masked_features = "./masked_feature/{}".format(self.opt.dataset)
        if not os.path.exists(out_masked_features):
            os.makedirs(out_masked_features)
        for task_id,task in enumerate(self.tasks):
            fname = os.path.join(self.data_dir,task)
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            # fin = open(fname+'.graph', 'rb')
            # idx2graph = pickle.load(fin)
            fin.close()
            all_data_dict = {}
            task_data = { polarity:[] for polarity in self.opt.polarities}
            out_masked_file = open(os.path.join(out_masked_features,task),'w',encoding='utf-8')
            aspect = task.split(".")[0].replace("#",' ').lower()
            aspect = aspect.replace("_"," ")
            for i in range(0, len(lines)):
                line  = lines[i].strip()
                line_data = line.split("\t")
                assert len(line_data)==2,"line data error"
                text = split_punc(line_data[0])
                masked_text, seed_words = self.get_masked_text(text,aspect)
                out_line = "{}\n{}\n{}\n".format(text, masked_text, seed_words)
                out_masked_file.write(out_line)
                polarity = line_data[1]
                text_indices = self.tokenizer.text_to_sequence(text)
                masked_indices = self.tokenizer.text_to_sequence(masked_text)
                seed_indices = self.tokenizer.text_to_sequence(seed_words)

                # right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
                # right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
                aspect_indices = self.tokenizer.text_to_sequence(aspect,max_seq_len=4)
                # left_len = np.sum(left_indices != 0)
                aspect_len = np.sum(aspect_indices != 0)
                # aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)
                if self.polarity2label.get(polarity) is not None:
                    label = self.polarity2label[polarity]
                else:
                    print("error polarity: {}\n{}".format(len(polarity),self.polarity2label))
                    label=None
                assert type(label)==int
                text_len = np.sum(text_indices != 0)
                concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")
                concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
                concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)

                text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
                # aspect_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")

                # dependency_graph = np.pad(idx2graph[i], \
                #     ((0,tokenizer.max_seq_len-idx2graph[i].shape[0]),(0,tokenizer.max_seq_len-idx2graph[i].shape[0])), 'constant')

                dependency_graph = None

                data = {
                    'concat_bert_indices': concat_bert_indices,
                    'concat_segments_indices': concat_segments_indices,
                    'text_bert_indices': text_bert_indices,
                    # 'aspect_bert_indices': aspect_bert_indices,
                    'text_indices': text_indices,
                    'masked_indices':masked_indices,
                    'seed_indices':seed_indices,
                    # 'context_indices': context_indices,
                    # 'left_indices': left_indices,
                    # 'left_with_aspect_indices': left_with_aspect_indices,
                    # 'right_indices': right_indices,
                    # 'right_with_aspect_indices': right_with_aspect_indices,
                    'aspect_indices': aspect_indices,
                    # 'all_aspect_indices':self.all_aspect_indices,
                    # 'aspect_boundary': aspect_boundary,
                    # 'dependency_graph': dependency_graph,
                    'task_id': task_id,
                    'polarity': label,
                }
                task_data[self.label2polarity[label]].append(data)
            out_masked_file.close()
            self.all_data[task] = task_data




class TaskDataset(Dataset):
    def __init__(self,data):
        self.data=data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# class FewshotDataLoader():
#     def __init__(self,all_task_dataset,type="train",support_size=5,query_size=15,shuffle=True,shuffle_on_task=False):
#         self.dataset = all_task_dataset
#         self.all_task_dataset = all_task_dataset.all_data
#         self.type = type
#         self.support_size = support_size
#         self.query_size = query_size
#         self.shuffle=shuffle
#         self.shuffle_on_task = shuffle_on_task
#         self.dataloaders= {}
#         self.is_task_empty = {}
#         self.polarities = self.dataset.polarities
#         self.ways = len(self.polarities)
#         self.all_task_name = list(self.all_task_dataset.keys())
#         self._build_dataloaders()
#         self.task_index = 0
#
#
#     def __len__(self):
#         length = 0
#         for key,loader in self.dataloaders.items():
#             if isinstance(loader,list):
#                 length += len(loader)
#             elif isinstance(loader,dict):
#                 tmp = 0
#                 for k,data in loader.items():
#                     tmp += len(data)
#                 length += (math.ceil(tmp/len(loader)))
#             else:
#                 print("data loader error")
#         return length
#
#
#
#     def _reset_loader_index(self):
#         for task_name,set_dict in self.loader_index:
#             for set_label,index in set_dict.items():
#                 self.loader_index[task_name][set_label] = 0
#
#     def reset_test_loader(self):
#         for k,v in self.loader_index.items():
#             self.loader_index[k] = 0
#         for k,v in self.is_task_empty.items():
#             self.is_task_empty[k]=False
#         self.task_index = 0
#
#
#     def _get_loader_index(self,task_name,set_label=None):
#         if self.type=="train":
#             current_index = self.loader_index[task_name][set_label]
#             if current_index == self.max_loader_index[task_name][set_label] -1:
#                 self.loader_index[task_name][set_label] = 0
#             else:
#                 self.loader_index[task_name][set_label] = current_index+1
#             return current_index
#         else:
#             current_index = self.loader_index[task_name]
#             if current_index == self.max_loader_index[task_name]:
#                 return -1
#             else:
#                 self.loader_index[task_name] = current_index + 1
#             return current_index
#
#     def get_task_name(self,random=False):
#         if random:
#             return random.sample(self.all_task_name, k=1)[0]
#         if self.task_index==len(self.all_task_name):
#             self.task_index = 1
#             return self.all_task_name[0]
#         else:
#             task_name = self.all_task_name[self.task_index]
#             self.task_index +=1
#             return task_name
#
#
#
#     def get_batch(self):
#         if self.type=="train":
#             current_task_name = self.get_task_name(self.shuffle_on_task)
#             pos_batch_index = self._get_loader_index(current_task_name,'pos')
#             neg_batch_index = self._get_loader_index(current_task_name,'neg')
#             pos_batch = deepcopy(self.dataloaders[current_task_name]['pos'][pos_batch_index])
#             neg_batch = deepcopy(self.dataloaders[current_task_name]['neg'][neg_batch_index])
#             batch_data = self._combine_batch(pos_batch,neg_batch)
#             return batch_data
#         else:
#             current_task_name = self.get_task_name()
#             query_batch_index = self._get_loader_index(current_task_name)
#             if query_batch_index == -1:
#                 self.is_task_empty[current_task_name] = True
#                 if not all(list(self.is_task_empty.values())):
#                     return self.get_batch()
#                 else:
#                     print(current_task_name)
#                     print("No Batch Left")
#                     return None
#             support_pos_batch = deepcopy(self.test_supprt_loaders[current_task_name]['pos'][query_batch_index])
#             support_neg_batch = deepcopy(self.test_supprt_loaders[current_task_name]['neg'][query_batch_index])
#             support_batch_data = self._combine_batch(support_pos_batch, support_neg_batch)
#             assert support_batch_data
#             query_batch_data = deepcopy(self.dataloaders[current_task_name][query_batch_index])
#             batch_data = {}
#             for data_key, support_tensor in support_batch_data.items():
#                 query_tensor = query_batch_data[data_key]
#                 batch_data[data_key] = torch.cat((support_tensor, query_tensor), dim=0)
#             return batch_data
#
#
#     def initiate_test_loader_with_random_support_set(self):
#         self.test_supprt_loaders = {}
#         for task_name, task_data in self.all_task_dataset.items():
#             self.dataloaders[task_name] = {}
#             self.test_supprt_loaders[task_name] = {}
#             task_left_sets = []
#             neg_set = task_data['neg']
#             pos_set = task_data['pos']
#             neg_index = [i for i in range(len(neg_set))]
#             pos_index = [i for i in range(len(neg_set))]
#             self.test_supprt_loaders[task_name]['pos'] = []
#             self.test_supprt_loaders[task_name]['neg'] = []
#             self.dataloaders[task_name] = []
#             sample_times = max(len(pos_set),len(neg_set))//self.support_size
#             for i in range(1,sample_times+1):
#                 sampled_neg_index = sample(neg_index,self.support_size)
#                 sampled_pos_index = sample(pos_index,self.support_size)
#                 left_neg_index = list(set(neg_index) - set(sampled_neg_index))
#                 left_pos_index = list(set(pos_index) - set(sampled_pos_index))
#                 print(sampled_neg_index)
#                 print(sampled_pos_index)
#                 sampled_neg_set = [neg_set[index] for index in sampled_neg_index]
#                 sampled_pos_set = [pos_set[index] for index in sampled_pos_index]
#                 neg_task_dataset = TaskDataset(sampled_neg_set)
#                 pos_task_dataset = TaskDataset(sampled_pos_set)
#                 self.test_supprt_loaders[task_name]['neg'].extend(list(DataLoader(neg_task_dataset, batch_size=self.support_size)))
#                 self.test_supprt_loaders[task_name]['pos'].extend(list(DataLoader(pos_task_dataset, batch_size=self.support_size)))
#                 task_left_sets = ([neg_set[index] for index in left_neg_index ]  + [ pos_set[index] for index in left_pos_index ] )
#                 sampled_query_set = sample(task_left_sets,self.query_size*2)
#                 self.dataloaders[task_name].extend(list(DataLoader(sampled_query_set,
#                                                               batch_size=self.query_size*2,
#                                                               shuffle=self.shuffle, drop_last=False)))
#             assert len(self.dataloaders[task_name]) > 0
#             self.max_loader_index[task_name] = len(self.dataloaders[task_name])
#             self.loader_index[task_name] = 0
#             self.is_task_empty[task_name] = False
#
#     # def initiate_test_loader_with_fix_support_set(self):
#     #     self.test_supprt_loaders = {}
#     #     for task_name, task_data in self.all_task_dataset.items():
#     #         self.dataloaders[task_name] = []
#     #         self.test_supprt_loaders[task_name] = {}
#     #         task_left_sets = []
#     #         for set_label, data_set in task_data.items():
#     #             support_set = TaskDataset(data_set[:self.support_size])
#     #             support_batch_data = list(
#     #                 DataLoader(support_set, batch_size=self.support_size))[0]
#     #             self.test_supprt_loaders[task_name][set_label] = list(
#     #                 DataLoader(support_set, batch_size=self.support_size))
#     #             task_left_sets += data_set[self.support_size:]
#     #         self.dataloaders[task_name] = list(DataLoader(task_left_sets,
#     #                                                       batch_size=self.query_size*2,
#     #                                                       shuffle=self.shuffle, drop_last=False))
#     #         assert len(self.dataloaders[task_name]) > 0
#     #         self.max_loader_index[task_name] = len(self.dataloaders[task_name])
#     #         self.loader_index[task_name] = 0
#     #         self.is_task_empty[task_name] = False
#     #     print(self.max_loader_index)
#
#
#
#     def initiate_test_loader_with_ten_random_support_set(self,times=10):
#         self.test_supprt_loaders = {}
#         self.max_test_loader_index = {}
#         for task_name, task_data in self.all_task_dataset.items():
#             self.dataloaders[task_name] = []
#             self.test_supprt_loaders[task_name] = {}
#             for set_label in task_data.keys():
#                 self.test_supprt_loaders[task_name][set_label] = []
#             for i in range(times):
#                 task_left_sets = []
#                 support_batches = {} ##sample different support batches for each time
#                 for set_label, data_set in task_data.items():
#                     indices = [i for i in range(len(data_set))]
#                     sampled_indices = sample(indices,self.support_size)
#                     left_indices = list(set(indices) - set(sampled_indices))
#                     support_set = TaskDataset([data_set[index] for index in sampled_indices])
#                     assert len(support_set)==self.support_size
#                     support_batches[set_label] = list(DataLoader(support_set, batch_size=self.support_size))[0]
#                     task_left_sets += [data_set[index] for index in left_indices]
#                 self.dataloaders[task_name].extend(list(DataLoader(task_left_sets,
#                                                           batch_size=self.query_size*2,
#                                                           shuffle=self.shuffle, drop_last=False)))
#                 for i in range(len(self.dataloaders[task_name])):
#                     for set_label,batch in support_batches.items():
#                         self.test_supprt_loaders[task_name][set_label].append(batch)
#             assert len(self.dataloaders[task_name]) > 0
#             self.max_loader_index[task_name] = len(self.dataloaders[task_name])
#             self.loader_index[task_name] = 0
#             self.is_task_empty[task_name] = False
#         print(self.max_loader_index)
#
#
#
#
#     def _combine_batch(self,pos_batch,neg_batch):
#         batch_dict = {}
#         for attri_name,pos_tensor in pos_batch.items():
#             neg_tensor = neg_batch[attri_name]
#             support_tensor = torch.cat([neg_tensor[:self.support_size],pos_tensor[:self.support_size] ],dim=0)
#             query_tensor = torch.cat([neg_tensor[self.support_size:],pos_tensor[self.support_size:]], dim=0)
#             final_batch = torch.cat([support_tensor,query_tensor],dim=0)
#             batch_dict[attri_name] = final_batch
#         return batch_dict
#
#
#     def _get_batch_list_from_loader(self,dataloader):
#         batch_list = []
#         for batch_idx, batch_data in enumerate(dataloader):
#             for k, v in batch_data.items():
#                 batch_length = len(v)
#                 break
#             if batch_length > self.support_size:
#                 batch_list.append((batch_data))
#         return batch_list
#
#
#     def _build_dataloaders(self):
#         self.loader_index = {}
#         self.max_loader_index = {}
#         if self.type=="train":
#             for task_name,task_data in self.all_task_dataset.items():
#                 self.dataloaders[task_name] = {}
#                 self.max_loader_index[task_name] = {}
#                 self.loader_index[task_name] = {}
#                 for set_label, set in task_data.items():
#                     task_set = TaskDataset(set)
#                     assert len(task_set) > self.support_size, "not enough example for query set"
#                     self.dataloaders[task_name][set_label] = self._get_batch_list_from_loader(DataLoader(task_set,batch_size=self.support_size+self.query_size,shuffle=self.shuffle,drop_last=False))
#                     assert len(self.dataloaders[task_name][set_label]) > 0, "{}_{} dataloader only have {} examles".format(task_name,set_label,len(task_set))
#                     self.max_loader_index[task_name][set_label] = len(self.dataloaders[task_name][set_label])
#                     self.loader_index[task_name][set_label] = 0
#         elif self.type=="dev":
#             self.initiate_test_loader_with_fix_support_set()
#         else :
#             self.initiate_test_loader_with_ten_random_support_set()

class AllAspectFewshotDataLoader():
    def __init__(self,all_task_dataset,type="train",support_size=5,query_size=15,shuffle=True,shuffle_on_task=False):
        self.dataset = all_task_dataset
        self.all_task_dataset = all_task_dataset.all_data
        self.type = type
        self.support_size = support_size
        self.query_size = query_size
        self.shuffle=shuffle
        self.shuffle_on_task = shuffle_on_task
        self.dataloaders= {}
        self.is_task_empty = {}
        self.polarities = self.dataset.polarities
        self.ways = len(self.polarities)
        self._build_dataloaders()
        self.all_task_name = list(self.all_task_dataset.keys())
        self.task_index = 0


    def __len__(self):
        length = 0
        length_list = []
        for key,loader in self.dataloaders.items():
            if isinstance(loader,list):
                length_list.append(len(loader))
            elif isinstance(loader,dict):
                tmp = 0
                for k,data in loader.items():
                    tmp += len(data)
                length += (math.ceil(tmp/len(loader)))
            else:
                print("data loader error")
        if len(length_list)>0:
            length = max(length_list)
        return length



    def _reset_loader_index(self):
        for task_name,set_dict in self.loader_index:
            for set_label,index in set_dict.items():
                self.loader_index[task_name][set_label] = 0

    def reset_test_loader(self):
        for k,v in self.loader_index.items():
            self.loader_index[k] = 0
        for k,v in self.is_task_empty.items():
            self.is_task_empty[k]=False
        self.task_index = 0


    def _get_loader_index(self,task_name,set_label=None):
        if self.type=="train":
            current_index = self.loader_index[task_name][set_label]
            if current_index == self.max_loader_index[task_name][set_label] -1:
                self.loader_index[task_name][set_label] = 0
            else:
                self.loader_index[task_name][set_label] = current_index+1
            return current_index
        else:
            current_index = self.loader_index[task_name]
            if current_index == self.max_loader_index[task_name]:
                self.loader_index[task_name] = 0
                return -1
            else:
                self.loader_index[task_name] = current_index + 1
            return current_index

    def get_task_name(self,random=False):
        if random:
            return random.sample(self.all_task_name, k=1)[0]
        if self.task_index==len(self.all_task_name):
            self.task_index = 1
            return self.all_task_name[0]
        else:
            task_name = self.all_task_name[self.task_index]
            self.task_index +=1
            return task_name



    def get_batch(self):
        if self.type=="train":
            pol_batchs = { polarity:[] for polarity in self.polarities}
            for polarity in self.polarities:
                for current_task_name in self.all_task_name:
                    pol_batch_index = self._get_loader_index(current_task_name,polarity)
                    pol_batchs[polarity].append(deepcopy(self.dataloaders[current_task_name][polarity][pol_batch_index]))
            batch_data = self._combine_batch_list(pol_batchs)
            return batch_data
        else:
            pol_batchs = {polarity:[] for polarity in self.polarities}
            query_batchs = []
            for current_task_name in self.all_task_name:
                query_batch_index = self._get_loader_index(current_task_name)
                for polarity in self.polarities:
                    pol_batchs[polarity].append(deepcopy(self.test_supprt_loaders[current_task_name][polarity][query_batch_index]))
                if query_batch_index == -1:
                    self.is_task_empty[current_task_name] = True
                    if not all(list(self.is_task_empty.values())):
                        pass
                    else:
                        print(current_task_name)
                        print("No Batch Left")
                        return None
                query_batchs.append(deepcopy(self.dataloaders[current_task_name][query_batch_index]))
            support_batch_data = self._combine_batch_list(pol_batchs)
            assert support_batch_data
            batch_data = {}
            for data_key, support_tensor in support_batch_data.items():
                query_tensors = [ query_batch[data_key] for query_batch in query_batchs if query_batch.get(data_key) is not None]
                if query_tensors: ## query tensors may be empty
                    query_tensor = torch.cat(query_tensors,dim=0)
                    batch_data[data_key] = torch.cat((support_tensor, query_tensor), dim=0)
                else:
                    batch_data[data_key] = support_tensor
            return batch_data


    # def initiate_test_loader_with_random_support_set(self):
    #     self.test_supprt_loaders = {}
    #     for task_name, task_data in self.all_task_dataset.items():
    #         self.dataloaders[task_name] = {}
    #         self.test_supprt_loaders[task_name] = {}
    #         task_left_sets = []
    #         neg_set = task_data['neg']
    #         pos_set = task_data['pos']
    #         neg_index = [i for i in range(len(neg_set))]
    #         pos_index = [i for i in range(len(neg_set))]
    #         self.test_supprt_loaders[task_name]['pos'] = []
    #         self.test_supprt_loaders[task_name]['neg'] = []
    #         self.dataloaders[task_name] = []
    #         sample_times = max(len(pos_set),len(neg_set))//self.support_size
    #         for i in range(1,sample_times+1):
    #             sampled_neg_index = sample(neg_index,self.support_size)
    #             sampled_pos_index = sample(pos_index,self.support_size)
    #             left_neg_index = list(set(neg_index) - set(sampled_neg_index))
    #             left_pos_index = list(set(pos_index) - set(sampled_pos_index))
    #             print(sampled_neg_index)
    #             print(sampled_pos_index)
    #             sampled_neg_set = [neg_set[index] for index in sampled_neg_index]
    #             sampled_pos_set = [pos_set[index] for index in sampled_pos_index]
    #             neg_task_dataset = TaskDataset(sampled_neg_set)
    #             pos_task_dataset = TaskDataset(sampled_pos_set)
    #             self.test_supprt_loaders[task_name]['neg'].extend(list(DataLoader(neg_task_dataset, batch_size=self.support_size)))
    #             self.test_supprt_loaders[task_name]['pos'].extend(list(DataLoader(pos_task_dataset, batch_size=self.support_size)))
    #             task_left_sets = ([neg_set[index] for index in left_neg_index ]  + [ pos_set[index] for index in left_pos_index ] )
    #             sampled_query_set = sample(task_left_sets,self.query_size*2)
    #             self.dataloaders[task_name].extend(list(DataLoader(sampled_query_set,
    #                                                           batch_size=self.query_size*2,
    #                                                           shuffle=self.shuffle, drop_last=False)))
    #         assert len(self.dataloaders[task_name]) > 0
    #         self.max_loader_index[task_name] = len(self.dataloaders[task_name])
    #         self.loader_index[task_name] = 0
    #         self.is_task_empty[task_name] = False

    # def initiate_test_loader_with_fix_support_set(self):
    #     self.test_supprt_loaders = {}
    #     for task_name, task_data in self.all_task_dataset.items():
    #         self.dataloaders[task_name] = []
    #         self.test_supprt_loaders[task_name] = {}
    #         task_left_sets = []
    #         for set_label, data_set in task_data.items():
    #             support_set = TaskDataset(data_set[:self.support_size])
    #             self.test_supprt_loaders[task_name][set_label] = list(
    #                 DataLoader(support_set, batch_size=self.support_size))
    #             task_left_sets += data_set[self.support_size:]
    #         self.dataloaders[task_name] = list(DataLoader(task_left_sets,
    #                                                       batch_size=self.query_size*2,
    #                                                       shuffle=self.shuffle, drop_last=False))
    #         assert len(self.dataloaders[task_name]) > 0
    #         self.max_loader_index[task_name] = len(self.dataloaders[task_name])
    #         self.loader_index[task_name] = 0
    #         self.is_task_empty[task_name] = False
    #     print(self.max_loader_index)



    def initiate_test_loader_with_ten_random_support_set(self,times=10):
        self.test_supprt_loaders = {}
        for task_name, task_data in self.all_task_dataset.items():
            self.dataloaders[task_name] = []
            self.test_supprt_loaders[task_name] = {}
            for set_label in task_data.keys():
                self.test_supprt_loaders[task_name][set_label] = []
            for i in range(times):
                task_left_sets = []
                support_batches = {} ##sample different support batches for each time
                for set_label, data_set in task_data.items():
                    indices = [i for i in range(len(data_set))]
                    sampled_indices = sample(indices,self.support_size)
                    left_indices = list(set(indices) - set(sampled_indices))
                    support_set = TaskDataset([data_set[index] for index in sampled_indices])
                    assert len(support_set)==self.support_size
                    support_batches[set_label] = list(DataLoader(support_set, batch_size=self.support_size))[0]
                    task_left_sets += [data_set[index] for index in left_indices]
                left_task_data = list(DataLoader(task_left_sets,batch_size=self.query_size*len(self.polarities),shuffle=self.shuffle, drop_last=False))
                self.dataloaders[task_name].extend(left_task_data)
                for i in range(len(left_task_data)):
                    for set_label,batch in support_batches.items():
                        self.test_supprt_loaders[task_name][set_label].append(batch)
            assert len(self.dataloaders[task_name]) > 0
            self.max_loader_index[task_name] = len(self.dataloaders[task_name])
            self.loader_index[task_name] = 0
            self.is_task_empty[task_name] = False
            for task, task_data_dict in self.test_supprt_loaders.items():
                for set_label, pol_data_list in task_data_dict.items():
                    assert len(pol_data_list)==len(self.dataloaders[task]),"test support and query set not equal: {} {}".format(len(pol_data_list),len(self.dataloaders[task]))
        print(self.max_loader_index)


    def _combine_batch_list(self,pol_batchs):
        batch_dict ={}
        pos_batch = pol_batchs[self.polarities[0]][0]
        attri_names = list(pos_batch.keys())
        for name in attri_names:
            support_pol_tensor_dict = {polarity:[] for polarity in self.polarities}
            query_pol_tensor_dict = {polarity:[] for polarity in self.polarities}
            for polarity in self.polarities:
                pol_batch_list = pol_batchs[polarity]
                support_pol_tensor_list = []
                query_pol_tensor_list = []
                for idx, pol_batch in enumerate(pol_batch_list):
                    pol_tensor = pol_batch[name]
                    support_pol_tensor_list.append(pol_tensor[:self.support_size])
                    query_pol_tensor_list.append(pol_tensor[self.support_size:])
                support_pol_tensor_dict[polarity] = torch.cat(support_pol_tensor_list,dim=0)
                query_pol_tensor_dict[polarity] = torch.cat(query_pol_tensor_list,dim=0)
            support_all_task_tensor = torch.cat([ support_pol_tensor_dict[polarity] for polarity in self.polarities],dim=0)
            query_all_task_tensor = torch.cat([query_pol_tensor_dict[polarity] for polarity in self.polarities],dim = 0)
            batch_dict[name] = torch.cat([support_all_task_tensor,query_all_task_tensor],dim=0)
            supportset_size = support_all_task_tensor.shape[0]
        batch_dict['supportset_size'] = torch.tensor(supportset_size)
        return batch_dict


    def _get_batch_list_from_loader(self,dataloader):
        batch_list = []
        for batch_idx, batch_data in enumerate(dataloader):
            for k, v in batch_data.items():
                batch_length = len(v)
                break
            if batch_length > self.support_size:
                batch_list.append((batch_data))
        return batch_list


    def _build_dataloaders(self):
        self.loader_index = {}
        self.max_loader_index = {}
        if self.type=="train":
            for task_name,task_data in self.all_task_dataset.items():
                self.dataloaders[task_name] = {}
                self.max_loader_index[task_name] = {}
                self.loader_index[task_name] = {}
                for set_label, set in task_data.items():
                    task_set = TaskDataset(set)
                    assert len(task_set) > self.support_size, "not enough example for query set"
                    self.dataloaders[task_name][set_label] = self._get_batch_list_from_loader(DataLoader(task_set,batch_size=self.support_size+self.query_size,shuffle=self.shuffle,drop_last=False))
                    assert len(self.dataloaders[task_name][set_label]) > 0, "{}_{} dataloader only have {} examles".format(task_name,set_label,len(task_set))
                    self.max_loader_index[task_name][set_label] = len(self.dataloaders[task_name][set_label])
                    self.loader_index[task_name][set_label] = 0
        else :
            self.initiate_test_loader_with_ten_random_support_set()

def _get_tasks(task_path):
    tasks = []
    with open(task_path) as file:
        for line in file.readlines():
            line = line.strip()
            tasks.append(line)
    return tasks

def _get_file_names(data_dir,tasks):
    return [ os.path.join(data_dir,task) for task in tasks]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__=="__main__":
    # config
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='induction', type=str, required=False)
    parser.add_argument('--dataset', default='fewshot_rest_3way_1', type=str, help='fewshot_rest,fewshot_mams',
                        required=False)
    parser.add_argument('--output_par_dir', default='test_outputs', type=str)
    parser.add_argument('--polarities', default=["positive", "neutral", "negative"], nargs='+',
                        help="if just two polarity switch to ['positive', 'negtive']", required=False)
    parser.add_argument('--optimizer', default='adam', type=str, required=False)
    parser.add_argument('--criterion', default='origin', type=str, required=False)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, required=False)
    parser.add_argument('--lr', default=5e-5, type=float, help='try 5e-5, 2e-5, 1e-3 for others', required=False)
    parser.add_argument('--dropout', default=0.1, type=float, required=False)
    parser.add_argument('--l2reg', default=0.01, type=float, required=False)
    parser.add_argument('--num_episode', default=500, type=int, help='try larger number for non-BERT models',
                        required=False)
    parser.add_argument('--log_step', default=10, type=int, required=False)
    parser.add_argument('--dev_interval', default=100, type=int, required=False)
    parser.add_argument('--log_path', default="./log", type=str, required=False)
    parser.add_argument('--embed_dim', default=300, type=int, required=False)
    parser.add_argument('--hidden_dim', default=128, type=int, required=False, help="lstm encoder hidden size")
    parser.add_argument('--feature_dim', default=2 * 128, type=int, required=False,
                        help="feature dim after encoder depends on encoder")
    parser.add_argument('--output_dim', default=64, type=int, required=False)
    parser.add_argument('--relation_dim', default=100, type=int, required=False)
    parser.add_argument('--bert_dim', default=768, type=int, required=False)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str, required=False)
    parser.add_argument('--max_seq_len', default=85, type=int, required=False)
    parser.add_argument('--shots', default=5, type=int, required=False, help="set 1 shot; 5 shot; 10 shot")
    parser.add_argument('--query_size', default=10, type=int, required=False,
                        help="set 20 for 1-shot; 15 for 5-shot; 10 for 10-shot")
    parser.add_argument('--iterations', default=3, type=int, required=False)
    # parser.add_argument('--hops', default=3, type=int)
    parser.add_argument('--patience', default=3000, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0', required=False)
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    # The following parameters are only valid for the lcf-bert model
    parser.add_argument('--local_context_focus', default='cdm', type=str, help='local context focus mode, cdw or cdm')
    parser.add_argument('--SRD', default=3, type=int,
                        help='semantic-relative-distance, see the paper of LCF-BERT model')
    opt = parser.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # seed
    if opt.seed:
        set_seed(opt.seed)
    model_classes = {
        'induction': "FewShotInduction",
        'aspect': "AspectFewshot",
        'relation': "FewShotRelation",
        'cnn_relation': "CNNRelation",
        'aspect_relation': "AspectRelation",
        'atae-lstm': "ATAE_LSTM",
        'aspect_induction': "AspectAwareInduction",
    }
    input_features = {
        'induction': ['text_indices', 'supportset_size'],
        'relation': ['text_indices', 'supportset_size'],
        'cnn_relation': ['text_indices', 'supportset_size'],
        'aspect': ['text_indices', 'masked_indices', 'seed_indices', 'supportset_size'],
        'aspect_relation': ['text_indices', 'masked_indices', 'seed_indices', 'supportset_size'],
        'atae-lstm': ['text_indices', 'aspect_indices', 'supportset_size'],
        'aspect_induction': ['text_indices', 'aspect_indices', 'supportset_size'],
    }
    dataset_files = {
        'fewshot_mams_3way_1': {
            "data_dir": "./datasets/fewshot_mams_3way",
            'train': './tasks/mams/train_tasks',
            'val': "./tasks/mams/val_tasks",
            'test': './tasks/mams/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_3way_2': {
            "data_dir": "./datasets/fewshot_mams_3way",
            'train': './tasks/mams/train_tasks_2',
            'val': "./tasks/mams/val_tasks_2",
            'test': './tasks/mams/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_3way_3': {
            "data_dir": "./datasets/fewshot_mams_3way",
            'train': './tasks/mams/train_tasks_3',
            'val': "./tasks/mams/val_tasks_3",
            'test': './tasks/mams/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_3way_4': {
            "data_dir": "./datasets/fewshot_mams_3way",
            'train': './tasks/mams/train_tasks_4',
            'val': "./tasks/mams/val_tasks_4",
            'test': './tasks/mams/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_14rest_3way_1': {
            "data_dir": "./datasets/fewshot_14rest_3way",
            'train': './tasks/14rest/train_tasks',
            'val': "./tasks/14rest/val_tasks",
            'test': './tasks/14rest/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_14rest_3way_2': {
            "data_dir": "./datasets/fewshot_14rest_3way",
            'train': './tasks/14rest/train_tasks_2',
            'val': "./tasks/14rest/val_tasks_2",
            'test': './tasks/14rest/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_14rest_3way_3': {
            "data_dir": "./datasets/fewshot_14rest_3way",
            'train': './tasks/14rest/train_tasks_3',
            'val': "./tasks/14rest/val_tasks_3",
            'test': './tasks/14rest/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_14rest_3way_4': {
            "data_dir": "./datasets/fewshot_14rest_3way",
            'train': './tasks/14rest/train_tasks_4',
            'val': "./tasks/14rest/val_tasks_4",
            'test': './tasks/14rest/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_3way_1': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks',
            'val': "./tasks/rest_3way/val_tasks",
            'test': './tasks/rest_3way/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_3way_2': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks_2',
            'val': "./tasks/rest_3way/val_tasks_2",
            'test': './tasks/rest_3way/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_3way_3': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks_3',
            'val': "./tasks/rest_3way/val_tasks_3",
            'test': './tasks/rest_3way/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_3way_4': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks_4',
            'val': "./tasks/rest_3way/val_tasks_4",
            'test': './tasks/rest_3way/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_rest_3way_5': {
            "data_dir": "./datasets/fewshot_rest_3way",
            'train': './tasks/rest_3way/train_tasks_4',
            'val': "./tasks/rest_3way/val_tasks_4",
            'test': './tasks/rest_3way/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_lap_3way_1': {
            "data_dir": "./datasets/fewshot_lap_3way",
            'train': './tasks/lap_3way/train_tasks',
            'val': "./tasks/lap_3way/val_tasks",
            'test': './tasks/lap_3way/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_lap.csv',
                           './seed_words/seed_words_list_attributes_lap.csv'],
        }, 'fewshot_lap_3way_2': {
            "data_dir": "./datasets/fewshot_lap_3way",
            'train': './tasks/lap_3way/train_tasks_2',
            'val': "./tasks/lap_3way/val_tasks_2",
            'test': './tasks/lap_3way/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_lap.csv',
                           './seed_words/seed_words_list_attributes_lap.csv'],
        },
        'fewshot_lap_3way_3': {
            "data_dir": "./datasets/fewshot_lap_3way",
            'train': './tasks/lap_3way/train_tasks_3',
            'val': "./tasks/lap_3way/val_tasks_3",
            'test': './tasks/lap_3way/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_lap.csv',
                           './seed_words/seed_words_list_attributes_lap.csv'],
        }, 'fewshot_lap_3way_4': {
            "data_dir": "./datasets/fewshot_lap_3way",
            'train': './tasks/lap_3way/train_tasks_4',
            'val': "./tasks/lap_3way/val_tasks_4",
            'test': './tasks/lap_3way/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_lap.csv',
                           './seed_words/seed_words_list_attributes_lap.csv'],
        },
        'fewshot_mams_2way_1': {
            "data_dir": "./datasets/fewshot_mams_2way",
            'train': './tasks/mams/train_tasks',
            'val': "./tasks/mams/val_tasks",
            'test': './tasks/mams/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_2way_2': {
            "data_dir": "./datasets/fewshot_mams_2way",
            'train': './tasks/mams/train_tasks_2',
            'val': "./tasks/mams/val_tasks_2",
            'test': './tasks/mams/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_2way_3': {
            "data_dir": "./datasets/fewshot_mams_2way",
            'train': './tasks/mams/train_tasks_3',
            'val': "./tasks/mams/val_tasks_3",
            'test': './tasks/mams/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_mams_2way_4': {
            "data_dir": "./datasets/fewshot_mams_2way",
            'train': './tasks/mams/train_tasks_4',
            'val': "./tasks/mams/val_tasks_4",
            'test': './tasks/mams/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        }, 'fewshot_rest_2way_1': {
            "data_dir": "./datasets/fewshot_rest_2way",
            'train': './tasks/rest_2way/train_tasks',
            'val': "./tasks/rest_2way/val_tasks",
            'test': './tasks/rest_2way/test_tasks',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_2way_2': {
            "data_dir": "./datasets/fewshot_rest_2way",
            'train': './tasks/rest_2way/train_tasks_2',
            'val': "./tasks/rest_2way/val_tasks_2",
            'test': './tasks/rest_2way/test_tasks_2',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_2way_3': {
            "data_dir": "./datasets/fewshot_rest_2way",
            'train': './tasks/rest_2way/train_tasks_3',
            'val': "./tasks/rest_2way/val_tasks_3",
            'test': './tasks/rest_2way/test_tasks_3',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
        'fewshot_rest_2way_4': {
            "data_dir": "./datasets/fewshot_rest_2way",
            'train': './tasks/rest_2way/train_tasks_4',
            'val': "./tasks/rest_2way/val_tasks_4",
            'test': './tasks/rest_2way/test_tasks_4',
            'seed_files': ['./seed_words/seed_words_list_entities_rest.csv',
                           './seed_words/seed_words_list_attributes_rest.csv'],
        },
    }
    optimizers = {
        'adam': "optim.Adam",
    }
    criterions = {
        'origin': "Criterion",
        'ce': "CrossEntropyCriterion",
    }
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.model_class = model_classes[opt.model_name]
    opt.dataset_files = dataset_files[opt.dataset]
    opt.optim_class = optimizers[opt.optimizer]
    opt.criterion_class = criterions[opt.criterion]
    opt.input_features = input_features[opt.model_name]
    opt.output_dir = os.path.join(opt.output_par_dir, opt.model_name,
                                  opt.dataset)  ##get output directory to save results
    opt.ways = len(opt.polarities)
    train_tasks = _get_tasks(opt.dataset_files['train'])
    val_tasks = _get_tasks(opt.dataset_files['val'])
    test_tasks = _get_tasks(opt.dataset_files['test'])
    train_fnames = _get_file_names(opt.dataset_files['data_dir'], train_tasks)
    val_fnames = _get_file_names(opt.dataset_files['data_dir'], val_tasks)
    test_fnames = _get_file_names(opt.dataset_files['data_dir'], test_tasks)

    tokenizer = build_tokenizer(
        fnames=train_fnames + val_fnames + test_fnames,
        max_seq_len=opt.max_seq_len,
        dat_fname='{0}_tokenizer.dat'.format(opt.dataset))
    embedding_matrix = build_embedding_matrix(
        word2idx=tokenizer.word2idx,
        embed_dim=opt.embed_dim,
        dat_fname='{0}_{1}_embedding_matrix.dat'.format(str(300), opt.dataset))
    print("tokenizer length: ", len(tokenizer.idx2word))
    print("embedding_matrix shape: ", embedding_matrix.shape)
    print("using model: ", opt.model_name)
    print("running dataset: ", opt.dataset)
    print("output_dir: ", opt.output_dir)
    print("shots: ", opt.shots)
    print("ways: ", opt.ways)
    data_dir = opt.dataset_files['data_dir']
    trainset = FewshotACSADataset(data_dir=data_dir, tasks=train_tasks,
                                       tokenizer=tokenizer, opt=opt)
    valset = FewshotACSADataset(data_dir=data_dir, tasks=val_tasks,
                                     tokenizer=tokenizer, opt=opt)
    testset = FewshotACSADataset(data_dir=data_dir, tasks=test_tasks,
                                      tokenizer=tokenizer, opt=opt)

    train_loader = AllAspectFewshotDataLoader(valset,type='test',shuffle=False)
    # train_loader = FewshotDataLoader(data_set,type='test',shuffle=False)
    print(len(train_loader))
    print(train_loader.max_loader_index)
    all_targets = []
    for i in range(len(train_loader)):
        batch = train_loader.get_batch()
        text_indices = batch['text_indices'].tolist()
        task_ids = batch['task_id'].tolist()
        targets = batch['polarity'].tolist()
        all_targets.extend(targets[15:])
        seed_indices = batch['seed_indices'].tolist()
        supportset_size = batch['supportset_size'].tolist()
    label_counter = Counter(all_targets)
    train_loader.get_batch()

