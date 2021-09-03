#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:26:28 2020

@author: Tao Shi 
"""

import numpy as np
import pickle
from collections import defaultdict # defaultdict: 当出现keyError异常时自动调用默认的工厂方法

 
MAX_LENGTH = 50  # 超参数：对话中每一句话的最大长度 


class Dataloader:
    def __init__(self, mode = None): 
        try:
            assert(mode is not None)
        except AssertionError:
            print("请设置识别模式为'情绪识别'或'情感识别'")
            exit()
 
        self.mode = mode  # ‘情绪识别’模型或者‘情感分析’模式 
        self.max_length = MAX_LENGTH 
        
        file = open("/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/data/features/data_{}.p".
                    format(self.mode.lower()),"rb")
        x = pickle.load(file)
        ''' 
        x中共包含6个list，具体数据结构如下所示：  
        - data: 包含6个dict： 
             - text: 原始的语句  
             - split: 表示该语句是属于训练集、验证集还是测试集
             - y: 语句的情感标签或者情绪标签
             - dialog: 该语句是属于哪一个对话(Dialogue_ID)
             - utterance: 该语句在该对话中的Utterance_ID 
             - num_words: 该语句中包含的单词数量  
        - W: 采用Glove对原始文本进行Word Embedding得到的向量 
        - vocab: 数据集中的词汇 
        - word_idx_map: 将vocab中的单词和其对应的W一一映射 
        - max_sentence_length: 数据集中一句话包含的最大符号数量
        - label_index: 将情感标签/情绪标签和其对应的数字一一映射，譬如label_index['neutral'] = 0
        '''
        self.data, self.W, self.word_idx_map, self.vocab, _, self.label_index = x[0], x[1], x[2], x[3], x[4], x[5]
        self.num_classes = len(self.label_index)
        print("采用的标签为: ", self.label_index)
           
        # 把样本划分到训练集、验证集和测试集 
        self.train_data, self.val_data, self.test_data = {},{},{}
        for i in range(len(self.data)):
            utterance_id = self.data[i]['dialog'] + "_" + self.data[i]['utterance']
            sentence_word_indices = self.get_word_indices(self.data[i]['text'])
            label = self.label_index[self.data[i]['y']] # 样本的标签 
   
            if self.data[i]['split'] == "train":
                self.train_data[utterance_id] = (sentence_word_indices,label)
            elif self.data[i]['split'] == "val":
                self.val_data[utterance_id] = (sentence_word_indices,label)
            elif self.data[i]['split'] == "test":
                self.test_data[utterance_id] = (sentence_word_indices,label)

        # 每个dialogue ID所对应的所有utterance ID 
        self.train_dialogue_ids = self.get_dialogue_ids(self.train_data.keys()) 
        self.val_dialogue_ids = self.get_dialogue_ids(self.val_data.keys())
        self.test_dialogue_ids = self.get_dialogue_ids(self.test_data.keys())
        
        self.max_utterances = self.get_max_utterances(self.train_dialogue_ids, self.val_dialogue_ids, self.test_dialogue_ids)
         
     
    ''' 
    返回语句对应的索引
    '''
    def get_word_indices(self, data_text):
        length = len(data_text.split())
        return np.array([self.word_idx_map[word] for word in data_text.split()] + [0] * (self.max_length - length))[:self.max_length]

 
    ''' 
    返回由Dialogue_ID与其包含的所有Utterance_ID组成的dict 
    '''
    def get_dialogue_ids(self, keys):
        ids = defaultdict(list)
        for key in keys:
            ids[key.split("_")[0]].append(int(key.split("_")[1]))
        for ID, utts in ids.items():
            ids[ID]=[str(utt) for utt in sorted(utts)]
        return ids 
 
  
    ''' 
    返回在训练集、验证集和测试集中最长的对话所包含的语句数量 
    '''
    def get_max_utterances(self, train_ids, val_ids, test_ids):
        max_utterances_train = max([len(train_ids[key]) for key in train_ids.keys()])
        max_utterances_val = max([len(val_ids[key]) for key in val_ids.keys()])
        max_utterances_test = max([len(test_ids[key]) for key in test_ids.keys()])
        return np.max([max_utterances_train, max_utterances_val, max_utterances_test])
 

    '''
    返回情感标签/情绪标签的独热编码 
    '''
    def get_one_hot_encoding(self, label):
        label_arr = [0] * self.num_classes
        label_arr[label] = 1
        return label_arr[:]
 
 
    '''
    获取音频特征 
    '''
    def get_dialogue_audio_embeddings(self):
        key = list(self.train_audio_emb.keys())[0]
        pad = [0] * len(self.train_audio_emb[key])

        def get_audio_embeddings(dialogue_id, audio_emb):
            dialogue_audio = []
            for k in dialogue_id.keys():
                local_audio = []
                for utt in dialogue_id[k]:
                    try:
                        local_audio.append(audio_emb[k + "_"+str(utt)][:])
                    except:
                        print(k + "_" + str(utt))
                        local_audio.append(pad[:])
                for _ in range(self.max_utterances-len(local_audio)):
                    local_audio.append(pad[:])
                dialogue_audio.append(local_audio[:self.max_utterances])
            return np.array(dialogue_audio)

        self.train_dialogue_features = get_audio_embeddings(self.train_dialogue_ids, self.train_audio_emb)
        self.val_dialogue_features = get_audio_embeddings(self.val_dialogue_ids, self.val_audio_emb)
        self.test_dialogue_features = get_audio_embeddings(self.test_dialogue_ids, self.test_audio_emb)

 
    '''
    获取文本特征
    '''
    def get_dialogue_text_embeddings(self):
        key = list(self.train_data.keys())[0]
        pad = [0] * len(self.train_data[key][0])

        def get_text_embeddings(dialogue_id, local_data):
            dialogue_text = []
            for k in dialogue_id.keys():
                local_text = []
                for utt in dialogue_id[k]:
                    local_text.append(local_data[k + "_" + str(utt)][0][:])
                for _ in range(self.max_utterances - len(local_text)):
                    local_text.append(pad[:])
                dialogue_text.append(local_text[:self.max_utterances])
            return np.array(dialogue_text)

        self.train_dialogue_features = get_text_embeddings(self.train_dialogue_ids, self.train_data)
        self.val_dialogue_features = get_text_embeddings(self.val_dialogue_ids, self.val_data)
        self.test_dialogue_features = get_text_embeddings(self.test_dialogue_ids, self.test_data)
    
    
    '''
    获取文本+音频双模态的双模态特征 
    '''
    def get_dialogue_bimodal_embeddings(self):
        
        '''
        使用concatenation对音频模态和文本模态进行特征融合 
        '''
        def concatenate_fusion(ID, text, audio):
            bimodal = []
            for vid, utts in ID.items():
                bimodal.append(np.concatenate((text[vid], audio[vid]), axis = 1))
            return np.array(bimodal)
        
        text_path = "/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/data/features/text_{}.pkl".format(
            self.mode.lower())
        audio_path = "/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/data/features/audio_{}.pkl".format(
            self.mode.lower())

        train_text_x, val_text_x, test_text_x = pickle.load(open(text_path, "rb"), encoding = 'latin1')
        train_audio_x, val_audio_x, test_audio_x = pickle.load(open(audio_path, "rb"), encoding = 'latin1')

        self.train_dialogue_features = concatenate_fusion(self.train_dialogue_ids, train_text_x, train_audio_x)
        self.val_dialogue_features = concatenate_fusion(self.val_dialogue_ids, val_text_x, val_audio_x)
        self.test_dialogue_features = concatenate_fusion(self.test_dialogue_ids, test_text_x, test_audio_x)
   
     
    '''
    获取样本的标签 
    '''
    def get_dialogue_labels(self):
        
        def get_labels(ids, data):
            dialogue_label=[]
 
            for k, utts in ids.items():
                local_labels = []
                for utt in utts:
                    local_labels.append(self.get_one_hot_encoding(data[k + "_" + str(utt)][1]))
                for _ in range(self.max_utterances - len(local_labels)):
                    local_labels.append(self.get_one_hot_encoding(1))
                dialogue_label.append(local_labels[:self.max_utterances])
            return np.array(dialogue_label)

        self.train_dialogue_label = get_labels(self.train_dialogue_ids, self.train_data)
        self.val_dialogue_label = get_labels(self.val_dialogue_ids, self.val_data)
        self.test_dialogue_label = get_labels(self.test_dialogue_ids, self.test_data)

 
    '''
    获取对话的长度 
    '''
    def get_dialogue_lengths(self):
        self.train_dialogue_length, self.val_dialogue_length, self.test_dialogue_length = [], [], []
        for vid, utts in self.train_dialogue_ids.items():
            self.train_dialogue_length.append(len(utts))
        for vid, utts in self.val_dialogue_ids.items():
            self.val_dialogue_length.append(len(utts))
        for vid, utts in self.test_dialogue_ids.items():
            self.test_dialogue_length.append(len(utts))
 

    '''  
    创建train_dialogue_features、val_dialogue_features和test_dialogue_features
    对应的mask向量，用来排除padding后续带来的影响
    '''
    def get_masks(self):
        self.train_mask = np.zeros((len(self.train_dialogue_length), self.max_utterances), dtype='float')
        for i in range(len(self.train_dialogue_length)):
            self.train_mask[i,:self.train_dialogue_length[i]] = 1.0
        self.val_mask = np.zeros((len(self.val_dialogue_length), self.max_utterances), dtype='float')
        for i in range(len(self.val_dialogue_length)):
            self.val_mask[i,:self.val_dialogue_length[i]] = 1.0
        self.test_mask = np.zeros((len(self.test_dialogue_length), self.max_utterances), dtype='float')
        for i in range(len(self.test_dialogue_length)):
            self.test_mask[i,:self.test_dialogue_length[i]] = 1.0

 
    '''
    加载音频模态数据 
    '''
    def load_audio_data(self):
        path = "/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/data/features/audio_embeddings_feature_selection_{}.pkl".format(self.mode.lower())
        self.train_audio_emb, self.val_audio_emb, self.test_audio_emb = pickle.load(open(path,"rb")) # 预训练的不同模态的Embedding 
        
        self.get_dialogue_audio_embeddings()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()


    '''
    加载文本模态数据 
    '''
    def load_text_data(self):
        self.get_dialogue_text_embeddings()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks()


    ''' 
    加载音频+文本双模态数据 
    '''  
    def load_bimodal_data(self):
        self.get_dialogue_bimodal_embeddings()
        self.get_dialogue_lengths()
        self.get_dialogue_labels()
        self.get_masks() 



'''
测试
'''
if __name__ == "__main__":
    test = Dataloader('emotion')
    test.get_dialogue_text_embeddings()
    print(test.train_dialogue_features.shape)
    test.get_dialogue_labels()
    print(test.train_dialogue_label.shape) 
    
    
    
    
    
    
    
