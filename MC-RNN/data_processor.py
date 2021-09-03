#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:34:09 2020

@author: Tao Shi
""" 

import torch  
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle
import pandas as pd 
   
    
class MELDDataset(Dataset): 
    def __init__(self, path, num_classes, train = True):
        file = open(path, 'rb')
        features = pickle.load(file)
        '''
        x中共包含9个list，具体数据结构如下所示：  
        - video_ids: 一个dict, key是所有数据集中的Dialogue_ID, value是每个Dialogue_ID包含的所有的utterance ID
        - video_speakers: 一个dict, key是所有数据集中的Dialogue_ID, value是用one-hot encoding来表示的每一句话的说话人
        - video_emotion_labels: 一个dict, key是所有数据集中的Dialogue_ID, value是每个Dialogue_ID包含的所有话的情绪标签 
        - video_text_embedding: 采用Glove得到的文本特征向量 
        - video_audio_embedding: 采用OpenSMILE得到的音频特征向量
        - video_visual_embeddinng: 采用3D CNN提取得到的视觉特征向量
        - video_sentence: 所有数据集中每一句话的原始文本 
        - train_and_val_ids: 训练集和验证集的所有Dialogue_ID 
        - test_ids: 测试集中的所有Dialogue_ID
        - video_sentiment_labels: 一个dict, key是所有数据集中的Dialogue_ID, value是每个Dialogue_ID包含的所有情感标签
        '''
        if num_classes == 3: # 情感分类 
            self.video_ids, self.video_speakers, _, self.video_text_embedding, self.video_audio_embedding,\
                self.video_sentence, self.train_and_val_ids, self.test_ids, self.video_labels = features
        elif num_classes == 7: # 情绪分类
             self.video_ids, self.video_speakers, self.video_labels, self.video_text_embedding, self.video_audio_embedding,\
                 self.video_sentence, self.train_and_val_ids, self.test_ids, _ = features

        # 若是训练，则keys为训练集+验证集的Dialogue_IDs；否则为测试集的Dialogue_IDs     
        self.keys = [x for x in (self.train_and_val_ids if train else self.test_ids)]
             
        self.length = len(self.keys)
       
      
    '''
    返回用tensor进行表示的第index个对话   
    '''
    def __getitem__(self, index):
        video_id = self.keys[index]
        return torch.FloatTensor(self.video_text_embedding[video_id]), torch.FloatTensor(self.video_audio_embedding[video_id]),\
            torch.FloatTensor(self.video_speakers[video_id]), \
                torch.FloatTensor([1] * len(self.video_labels[video_id])), torch.LongTensor(self.video_labels[video_id]), video_id
    
    '''
    返回对话数量
    ''' 
    def __len__(self):
        return self.length

 
    '''
    实现自定义的数据读取：使用pad_squence函数将长度不同的序列补齐到统一的长度
    '''
    def collate_fn(self, data):
        d = pd.DataFrame(data)
        return [pad_sequence(d[i]) if i < 3 else pad_sequence(d[i], True) if i < 5 else d[i].tolist() for i in d]



if __name__ == "__main__":
    path = '/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/Multimodal_Conversational_RNN/features/MELD_features/MELD_features_raw.pkl'
    trainset = MELDDataset(path = path, num_classes = 3, train = False)
    print(len(trainset))


        
    