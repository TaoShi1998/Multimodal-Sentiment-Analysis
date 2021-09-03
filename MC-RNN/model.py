#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 10:37:20 2020

@author: Tao Shi
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import time
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import numpy as np
from data_processor import MELDDataset
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
import math

 
'''
实现全局注意力机制（Global Attention）
'''
class Attention(nn.Module): 

    def __init__(self, global_dimension, utterance_dimension, alpha_dimension = None, attribute_type = 'general'):
        super(Attention, self).__init__()

        self.global_dimension = global_dimension 
        self.utterance_dimension = utterance_dimension
        self.attribute_type = attribute_type 
        
        if attribute_type == 'concat':
            self.transform = nn.Linear(utterance_dimension + global_dimension, alpha_dimension, bias = False)
            self.vector_prod = nn.Linear(alpha_dimension, 1, bias = False)
        elif attribute_type == 'general':
            self.transform = nn.Linear(utterance_dimension, global_dimension, bias = False)
        else:
            self.transform = nn.Linear(utterance_dimension, global_dimension, bias = True)
   
      
    def forward(self, M, x, mask = None):
        '''
        - M: (sequence_length, batch_size, context_dimension)
        - x: (batch_size, utterance_dimension)
        - mask: (batch_size, sequence_length) 
        '''
        if type(mask) == type(None):
        # torch.zeros(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) → Tensor
        # Returns a tensor filled with the scalar value 1, with the shape defined by the variable argument size 
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())
             
        if self.attribute_type == 'concat':      # 连接
            M_ = M.transpose(0, 1)                                               # batch_size, sequence_length, global_dimension
            # expand(*sizes) → Tensor
            # Passing -1 as the size for a dimension means not changing the size of that dimension.
            # 将size为1的维度扩展到更大的维度 
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)                      # batch_size, sequence_length, utterance_dimension
            # torch.cat(tensors, dim=0, out=None) → Tensor
            # Concatenates the given sequence of seq tensors in the given dimension. 
            # All tensors must either have the same shape (except in the concatenating dimension) or be empty. 
            # dim (int, optional) – the dimension over which the tensors are concatenated 
            M_x_ = torch.cat([M_, x_], 2)                                        # batch_size, sequence_length, global_dimension + utterance_dimension
            mx_a = F.tanh(self.transform(M_x_))                                  # batch_size, sequence_length, alpha_dimennsion
            alpha = F.softmax(self.vector_prod(mx_a), 1)
            alpha = alpha.transpose(1, 2)                                        # batch_size, 1, sequence_length
        elif self.attribute_type == 'general':  # 一般形式
            M_ = M.permute(1, 2, 0)                                              # batch_size, global_dimension, sequence_length
            x_ = self.transform(x).unsqueeze(1)                                  # batch_size, 1, global_dimension
            alpha = F.softmax(torch.bmm(x_, M_), dim = 2)                        # batch_size, 1, sequence_length
        else:                                   # 内积
            M_ = M.permute(1, 2, 0)                                              # batch_size, vector, seqence_length
            # torch.unsqueeze(input, dim) → Tensor
            # dim (int) – the index at which to insert the singleton dimension 
            # Returns a new tensor with a dimension of size 1 inserted at the specified position.
            # Example: before it was (4,) and after it was (4, 1) (when second parameter is 1).  
            x_ = x.unsqueeze(1)                                                   # batch_size, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim = 2)                        # batch_size, 1, sequence_length

        attention_pool = torch.bmm(alpha, M.transpose(0, 1))
        attention_pool = attention_pool[:, 0, :]                                 # batch_size, global_dimension

        return attention_pool, alpha

 

'''
Multimodal Conversational Recurrent Neural Network的基本单元 
'''
class ConversationalRNNCell(nn.Module):

    def __init__(self, D_m, D_global, D_participant, D_sentiment, context_attention = 'general', D_alpha = 100, dropout = 0.1):
        super(ConversationalRNNCell, self).__init__()

        self.D_m = D_m
        self.D_global = D_global
        self.D_participant = D_participant
        self.D_sentiment = D_sentiment
        
        self.global_cell = nn.GRUCell(D_m + D_participant, D_global)      # 全局GRU
        self.participant_cell = nn.GRUCell(D_m + D_global, D_participant) # 参与者GRU 
        self.sentiment_cell = nn.GRUCell(D_participant, D_sentiment)      # 情感GRU
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(D_global, D_m, D_alpha, context_attention)
  
      
    ''' 
    返回当前对话所包含的说话人 
    '''
    def select_parties(self, X, indices): 
        speaker0_sel = []
        for idx, x in zip(indices, X):
            speaker0_sel.append(x[idx].unsqueeze(0))
        speaker0_sel = torch.cat(speaker0_sel, 0)
        return speaker0_sel 
  
     
    def forward(self, utterance, speaker_mask, global_history, speaker0, sentiment0):
        ''' 
        utterance -> batch_size, D_m
        speaker_mask -> batch_size, participant
        global_history -> t-1, batch_size, D_global
        speaker0 -> batch_size, participant, D_participant
        sentiment0 -> batch_size, D_sentiment
        '''    
        # 返回speaker_mask中每一行的最大值对应的下标   
        speaker_mask_id = torch.argmax(speaker_mask, 1)
        speaker0_sel = self.select_parties(speaker0, speaker_mask_id) 
        global_ = self.global_cell(torch.cat([utterance, speaker0_sel], dim = 1), torch.zeros(utterance.size()[0], self.D_global).type(utterance.type()) \
                                   if global_history.size()[0] == 0 else global_history[-1])
        global_ = self.dropout(global_)
         
        if global_history.size()[0] == 0:
            context_ = torch.zeros(utterance.size()[0], self.D_global).type(utterance.type())
            alpha = None
        else: 
            context_, alpha = self.attention(global_history, utterance)
        
        # 将第二维扩展为speaker_mask.size()[1]
        utterance_context_ = torch.cat([utterance, context_], dim = 1).unsqueeze(1).expand(-1,speaker_mask.size()[1], -1)
        speaker_state_ = self.participant_cell(utterance_context_.contiguous().view(-1, self.D_m + self.D_global), \
                                           speaker0.view(-1, self.D_participant)).view(utterance.size()[0], -1, self.D_speaker)
        speaker_state_ = self.dropout(speaker_state_)
        listener_ = speaker0                   # 其他参与者的状态更新
        speaker_mask_ = speaker_mask.unsqueeze(2)
        speaker_ = listener_ * (1 - speaker_mask_) + speaker_state_ * speaker_mask_
        sentiment0 = torch.zeros(speaker_mask.size()[0], self.D_sentiment).type(utterance.type()) if sentiment0.size()[0] == 0 else sentiment0
        sentiment_ = self.sentiment_cell(self.select_parties(speaker_, speaker_mask_id), sentiment0)
        sentiment_ = self.dropout(sentiment_) # 防止过拟合 

        return global_, speaker_, sentiment_, alpha 


 
'''
由基本单元构成的Multimodal Conversational Recurrent Neural Network  
'''

class ConversationalRNN(nn.Module):

    def __init__(self, D_m, D_global, D_participant, D_sentiment, context_attention = 'general', D_alpha = 100, dropout = 0.1):
        super(ConversationalRNN, self).__init__()

        self.D_m = D_m
        self.D_global = D_global
        self.D_participant = D_participant 
        self.D_sentiment = D_sentiment
        self.dropout = nn.Dropout(dropout)

        self.conversational_cell = ConversationalRNNCell(D_m, D_global, D_participant, D_sentiment, context_attention, D_alpha, dropout)

  
    def forward(self, utterance, speaker_mask): 
        '''
        utterance -> sequence_length, batch_size, D_m
        speaker_mask -> sequence_length, batch_size, party
        '''
        global_history = torch.zeros(0).type(utterance.type())                                                              # 0-dimensional tensor
        speaker_ = torch.zeros(speaker_mask.size()[1], speaker_mask.size()[2], self.D_participant).type(utterance.type())   # batch_size, party, D_participant
        sentiment_ = torch.zeros(0).type(utterance.type())                                                                  # batch_size, D_sentiment
        sentiment = sentiment_
        
        alpha = []
        for utterance_, speaker_mask_ in zip(utterance, speaker_mask):
            global_, speaker_, sentiment_, alpha_ = self.conversational_cell(utterance_, speaker_mask_, global_history, speaker_, sentiment_)
            global_history = torch.cat([global_history, global_.unsqueeze(0)], 0)
            sentiment = torch.cat([sentiment, sentiment_.unsqueeze(0)], 0)
            if type(alpha_) != type(None):
                alpha.append(alpha_[:, 0, :])  # 第二维的元素
 
        return sentiment, alpha                  # sequence_length, batch, D_sentiment
 
 
      
'''
由Multimodal Conversational Recurrent Neural Network构成的主模型
'''
class Model(nn.Module):
 
    def __init__(self, D_m, D_global, D_participant, D_sentiment, D_h, num_classes, context_attention = 'general', 
                 D_alpha = 100, dropout_rec = 0.1, dropout = 0.1):
        super(Model, self).__init__()
        
        self.D_m = D_m
        self.D_global = D_global
        self.D_participant = D_participant
        self.D_sentimenr = D_sentiment
        self.D_h = D_h
        self.num_classes = num_classes 
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout + 0.15)  
        
        self.conversational_rnn = ConversationalRNN(D_m, D_global, D_participant, D_sentiment, context_attention, D_alpha, dropout_rec)
        self.linear = nn.Linear(D_sentiment, D_h)
        self.softmax = nn.Linear(D_h, num_classes)
        self.attention = Attention(D_sentiment, D_sentiment, attribute_type = 'general')
 

    def forward(self, utterance, speaker_mask, utterance_mask = None, set_attention = False):
        '''
        utterance -> sequence_length, batch_size, D_m
        #speaker_mask -> sequence_length, batch_size, party
        '''
        sentiments = self.conversational_rnn(utterance, speaker_mask)    # sequence_length, batch, D_sentiment
        sentiments = self.dropout_rec(sentiments)
        
        if set_attention:
            att_sentiments = []
            for e in sentiments:
                att_sentiments.append(self.attention(sentiments, e, mask = utterance_mask)[0].unsqueeze(0))
            att_sentiments = torch.cat(att_sentiments, dim = 0)
            # relu: 线性整流函数
            hidden = F.relu(self.linear(att_sentiments))
        else:
            hidden = F.relu(self.linear(sentiments))
            
        hidden = self.dropout(hidden)
        # log_softmax: mathematically equivalent to log(softmax(x))
        log_prob = F.log_softmax(self.softmax(hidden), 2)               # sequence_length, batch_size, num_classes
        
        return log_prob     
 
         
     
'''
由两个Multimodal Conversational Recurrent Neural Network构成的双向主模型 
'''
class BiModel(nn.Module):

    def __init__(self, D_m, D_global, D_participant, D_sentiment, D_h, num_classes, context_attention = 'general', 
                 D_alpha = 100, dropout_rec = 0.1, dropout = 0.1):
        super(BiModel, self).__init__()

        self.D_m = D_m
        self.D_global = D_global 
        self.D_participant = D_participant
        self.D_sentiment = D_sentiment 
        self.D_h = D_h 
        self.num_classes = num_classes 
        self.dropout = nn.Dropout(dropout)
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.conversational_rnn_forward = ConversationalRNN(D_m, D_global, D_participant, D_sentiment, context_attention, D_alpha, dropout_rec)
        self.conversational_rnn_backwards = ConversationalRNN(D_m, D_global, D_participant, D_sentiment, context_attention, D_alpha, dropout_rec)
        self.linear = nn.Linear(2 * D_sentiment, 2 * D_h)
        self.smax_fc = nn.Linear(2 * D_h, num_classes)
        self.attention = Attention(2 * D_sentiment, 2 * D_sentiment, attribute_type = 'general')
 

    def reverse_sequence(self, X, mask):
        '''
        X -> sequence_length, batch_size, dimension
        mask -> batch_size, sequence_length
        '''
        X_ = X.transpose(0, 1) 
        mask_sum = torch.sum(mask, 1).int()
        xfs = []
        for x, c in zip(X_, mask_sum): 
            # torch.flip(input, dims) → Tensor
            # Reverse the order of a n-D tensor along given axis in dims.
            # dims (a list or tuple) – axis to flip on
            xf = torch.flip(x[:c], [0])
            xfs.append(xf)

        return pad_sequence(xfs)

  
    def forward(self, utterance, speaker_mask, utterance_mask, set_attention = True):
        '''
        utterance -> sequence_length, batch_size, D_m
        speaker_mask -> sequence_length, batch_size, party
        '''
        sentiments_forward, alpha_forward = self.conversational_rnn_forward(utterance, speaker_mask)   # sequence_length, batch_size, D_sentiment
        sentiments_forward = self.dropout_rec(sentiments_forward)
        reverse_utterance = self.reverse_sequence(utterance, utterance_mask)
        reverse_speaker_mask = self.reverse_sequence(speaker_mask, utterance_mask)
        sentiment_backwards, alpha_backwards = self.conversational_rnn_backwards(reverse_utterance, reverse_speaker_mask)
        sentiment_backwards = self.reverse_sequence(sentiment_backwards, utterance_mask)
        sentiment_backwards = self.dropout_rec(sentiment_backwards)
        sentiments = torch.cat([sentiment_backwards, sentiments_forward], dim = -1)
         
        if set_attention:
            att_sentiments = []
            alpha = [] 
            for s in sentiments:
                att_sen, alpha_ = self.attention(sentiments, s, mask = utterance_mask)
                att_sentiments.append(att_sen.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_sentiments = torch.cat(att_sentiments, dim = 0)
            hidden = F.relu(self.linear(att_sentiments))
        else:
            hidden = F.relu(self.linear(sentiments))
            
        hidden = self.dropout(hidden)
        log_prob = F.log_softmax(self.smax_fc(hidden), 2)                                           # sequence_length, batch_size, num_classes
        
        return log_prob, alpha, alpha_forward, alpha_backwards
  


'''
计算负对数似然损失 (Negative Log Likelihood Loss, NLLLoss)
'''
class MaskedNLLLoss(nn.Module):

    def __init__(self, weight = None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        # torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')
        # reduction (string, optional) – Specifies the reduction to apply to the output: 'sum': the output will be summed. 
        self.loss = nn.NLLLoss(weight = weight, reduction = 'sum')


    def forward(self, predict, target, mask):
        '''
        predict -> batch_size * sequence_length, num_classes
        target -> batch_size * sequence_length
        mask -> batch_size, sequence_length
        '''
        mask_ = mask.view(-1, 1)       # batch_size * sequence_length, 1
        if type(self.weight) == type(None):
            loss = self.loss(predict * mask_, target) / torch.sum(mask)
        else:
            # torch.squeeze(input, dim=None, out=None) → Tensor
            # Returns a tensor with all the dimensions of input of size 1 removed.
            # For example, if input is of shape: (A×1×B×C×1×D) then the out tensor will be of shape: (A×B×C×D) .
            loss = self.loss(predict * mask_, target) / torch.sum(self.weight[target] * mask_.squeeze())
            
        return loss
    
 
 
'''
在MELD数据集上训练和评估模型
''' 
class TrainEvaluateMELD:
    
    def __init__(self, classification_mode, modality, dialogueID):
        self.classification_mode = classification_mode   # 情绪识别或者情感识别 
        self.modality = modality              # 单模态(文本或者音频)或双模态(文本+音频) 
        self.model_type = 'bimodel'           # 使用Conversational RNN or Bidirectional Conversational RNN
        self.dialogueID = dialogueID
        self.PATH = "/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/data/weights/{}_weights_{}.hdf5".format(modality, self.classification_mode)
        print("分类任务为: {} Recognition".format(self.classification_mode.capitalize()))
        print('模态信息为: {}'.format(self.modality.capitalize()))
     
     
    '''
    加载数据
    '''
    def load_data(self):
        if self.classification_mode == 'emotion':
            self.num_classes = 7
            self.label_names = ['neutral', 'surprise', 'fear', 'sadness', 'joy', 'disgust', 'anger']
            self.class_weights = torch.FloatTensor([4.0, 15.0, 15.0, 3.0, 1.0, 6.0, 3.0])
        else:
            self.num_classes = 3
            self.label_names = ['neutral', 'positive', 'negative']
            self.class_weights = torch.FloatTensor([1.0, 2.4, 1.5])

        if self.modality == 'text':
            self.D_m = 600
        elif self.modality == 'audio':
            self.D_m = 300
        elif self.modality == 'bimodal':
            self.D_m = 900
        else:
            self.D_m = 1200
            
        self.D_global = 150
        self.D_speaker = 150
        self.D_emotion = 100
        self.D_h = 100
        
        data_path = '/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/Multimodal_Conversational_RNN/features/MELD_features/MELD_features_raw.pkl'
        self.train_loader, self.valid_loader, self.test_loader = self.load_dataset(data_path, self.num_classes)
    
    
    '''
    划分训练集和验证集 
    '''
    def divide_train_valid(self, trainset, valid = 0.1):
        size = len(trainset)
        idx = list(range(size))
        split = int(valid * size) 

        return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])
    
    
    '''  
    加载MELD数据集到训练集、验证集和测试集 
    '''
    # torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, 
    # batch_sampler=None, num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, 
    # timeout=0, worker_init_fn=None, multiprocessing_context=None)
    # sampler (Sampler, optional) – defines the strategy to draw samples from the dataset. If specified, shuffle must be False.
    # num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    # collate_fn (callable, optional) – merges a list of samples to form a mini-batch of Tensor(s). Used when using batched loading from a map-style dataset.
    # pin_memory (bool, optional) – If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    def load_dataset(self, path, num_classes, batch_size = 32, valid = 0.1, num_workers = 0, pin_memory = False):
        trainset = MELDDataset(path, num_classes)
        train_sampler, valid_sampler = self.divide_train_valid(trainset, valid) 
        testset = MELDDataset(path, num_classes, train = False)
        
        train_loader = DataLoader(trainset,
                                  batch_size = batch_size,
                                  sampler = train_sampler,
                                  num_workers = num_workers,
                                  collate_fn = trainset.collate_fn,
                                  pin_memory = pin_memory)
        
        valid_loader = DataLoader(trainset,
                                  batch_size = batch_size,
                                  sampler = valid_sampler,
                                  num_workers = num_workers,
                                  collate_fn = trainset.collate_fn,                        
                                  pin_memory = pin_memory)
        
        test_loader = DataLoader(testset,
                                 batch_size = batch_size,
                                 num_workers = num_workers,
                                 collate_fn = testset.collate_fn,
                                 pin_memory = pin_memory)
         
        return train_loader, valid_loader, test_loader
    
    
    
    ''' 
    模型训练 
    ''' 
    def train(self):
        if self.model_type == 'model': 
            print('创建主模型')
            model = Model(self.D_m, self.D_global, 
                          self.D_speaker, self.D_emotion, 
                          self.D_h, self.num_classes)
        else:
            print('创建双向主模型')
            model = BiModel(self.D_m, self.D_global, 
                          self.D_speaker, self.D_emotion, 
                          self.D_h, self.num_classes)
        
        loss_function  = MaskedNLLLoss(self.class_weights)
        optimizer = optim.Adam(model.parameters(), lr = 0.0005, weight_decay = 0.00001)  
        
        self.best_fscore, self.best_loss, self.best_label, self.best_predict, self.best_mask = None, None, None, None, None
        self.train_losses, self.train_accuracies, self.valid_losses, self.valid_accuracies = [], [], [], []
        best_valid_loss = 1000.0
        count = 0
        epoch = 100 
         
        for e in range(epoch): 
            start_time = time.time() 
            train_loss, train_acc, _, _, _, train_fscore, _, _, _ = self.train_or_eval_model(model, loss_function, self.train_loader, e, optimizer, True)
            valid_loss, valid_acc, _, _, _, val_fscore, _, _ , _= self.train_or_eval_model(model, loss_function, self.valid_loader, e)
            test_loss, test_acc, test_label, test_predict, test_mask, test_fscore, attentions, test_class_report, test_class_report_dict = \
                self.train_or_eval_model(model, loss_function, self.test_loader, e)
             
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.valid_losses.append(valid_loss)
            self.valid_accuracies.append(valid_acc) 
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                count = 0
            else:
                count += 1
            
            if count == 20:
                epoch = e + 1
                break 
             
            print('current test_fscore: {}'.format(test_fscore))
            print('latest best_test_fscore: {}'.format(self.best_fscore))
            
            if self.classification_mode == 'sentiment':
                if self.best_fscore == None or self.best_fscore < test_fscore:
                    self.best_fscore, self.best_loss, self.best_label, self.best_predict, self.best_mask, self.test_class_report = test_fscore, \
                        test_loss, test_label, test_predict, test_mask, test_class_report
            else:
                test_neutral_fscore = test_class_report_dict[self.label_names[0]]['f1-score']
                test_surprise_fscore = test_class_report_dict[self.label_names[1]]['f1-score']
                test_fear_fscore = test_class_report_dict[self.label_names[2]]['f1-score']
                test_sadness_fscore = test_class_report_dict[self.label_names[3]]['f1-score']
                test_joy_fscore = test_class_report_dict[self.label_names[4]]['f1-score']
                test_disgust_fscore = test_class_report_dict[self.label_names[5]]['f1-score']
                test_anger_fscore = test_class_report_dict[self.label_names[6]]['f1-score']
                
                if self.best_fscore == None or (self.best_fscore < test_fscore and test_neutral_fscore > 0 and test_surprise_fscore > 0 and test_fear_fscore > 0 \
                                                and test_sadness_fscore and test_joy_fscore > 0 and test_disgust_fscore > 0 and test_anger_fscore > 0):
                    self.best_fscore, self.best_loss, self.best_label, self.best_predict, self.best_mask, self.test_class_report = test_fscore, test_loss, \
                        test_label, test_predict, test_mask, test_class_report
            
            print('epoch {}: train loss: {}, train accuracy: {}%, train f1-score: {:5.3f}, validation loss: {}, validation accuracy: {}%, \
                  validation f1-score: {:5.3f}, test loss: {}, test accuracy: {}%, test f1-score: {:5.3f}, time: {}s'.\
            format(e + 1, train_loss, train_acc, train_fscore / 100, valid_loss, valid_acc, val_fscore / 100, test_loss, test_acc, \
                   test_fscore / 100, round(time.time() - start_time, 2)))
        
        train_data = [self.best_fscore / 100, self.best_loss, self.best_label, self.best_predict, self.best_mask, \
                      self.test_class_report, self.train_losses, self.train_accuracies, self.valid_losses, self.valid_accuracies, epoch]
        np.save('{}_{}_{}'.format(self.model_type, self.classification_mode, self.modality), train_data)
        
    
    
    '''
    训练模型或者评估模型  
    '''
    def train_or_eval_model(self, model, loss_function, dataloader, epoch, optimizer = None, train = False):
        losses = []
        predicts = []
        labels = []
        masks = []
        alphas, alphas_forward, alphas_backwards, vids = [], [], [], []
        
        assert not train or optimizer != None
        
        if train:
            model.train()
        else: 
            model.eval()
         
        for data in dataloader: 
            if train:
                optimizer.zero_grad() 
                
            text_forward, acoustic_forward, visual_forward, speaker_mask, utterance_mask, label = data[:-1]
            
            if self.modality == "text":        # 文本模态
                log_prob, alpha, alpha_forward, alpha_backwards = model(text_forward, speaker_mask,utterance_mask)       # sequence_length, batch_size, num_classes
            elif self.modality == "audio":     # 音频模态
                log_prob, alpha, alpha_forward, alpha_backwards = model(acoustic_forward, speaker_mask,utterance_mask)   # sequence_length, batch_size, num_classes
            elif self.modality == 'bimodal':   # 文本+音频双模态
                log_prob, alpha, alpha_forward, alpha_backwards = model(torch.cat((text_forward, acoustic_forward),dim = -1), speaker_mask, utterance_mask)   # sequence_len, batch_size, num_classes
            else:                              # 文本+音频+视觉多模态 
                log_prob, alpha, alpha_forward, alpha_backwards = model(torch.cat((text_forward, acoustic_forward, visual_forward),dim = -1), speaker_mask, utterance_mask)
            
            lp_ = log_prob.transpose(0, 1).contiguous().view(-1, log_prob.size()[2])                                     # batch_size * sequence_length, num_classes
            labels_ = label.view(-1)                 # batch_size * sequence_length
            loss = loss_function(lp_, labels_, utterance_mask)
            predicts_ = torch.argmax(lp_, 1)         # batch_size * sequence_length
            predicts.append(predicts_.data.cpu().numpy())
            # cpu(): Moves all model parameters and buffers to the CPU
            labels.append(labels_.data.cpu().numpy()) 
            masks.append(utterance_mask.view(-1).cpu().numpy())
            losses.append(loss.item() * masks[-1].sum())
            
            if train: 
                loss.backward()
                optimizer.step()
            else: 
                alphas += alpha
                alphas_forward += alpha_forward
                alphas_backwards += alpha_backwards
                vids += data[-1]
         
        if predicts != []:
            # numpy.concatenate((a1, a2, ...), axis=0, out=None): join a sequence of arrays along an existing axis.
            predicts  = np.concatenate(predicts)
            labels = np.concatenate(labels)
            masks  = np.concatenate(masks)
        else:
            return float('nan'), float('nan'), [], [], [], float('nan'), []
        
        average_loss = round(np.sum(losses) / np.sum(masks), 4)
        average_accuracy = round(accuracy_score(labels, predicts, sample_weight = masks) * 100, 2)
        average_fscore = round(f1_score(labels, predicts, sample_weight = masks, average = 'weighted') * 100, 2)
        class_report = classification_report(labels, predicts, target_names = self.label_names, sample_weight = masks, digits = 3)
        class_report_with_dict = classification_report(labels, predicts, target_names = self.label_names, sample_weight = masks,digits = 3, output_dict = True)
        
        data = [predicts, labels]
        np.save('{}_{}_{}_prediction.npy'.format(self.model_type, self.classification_mode, self.modality), data)
        
        return average_loss, average_accuracy, labels, predicts, masks, average_fscore, [alphas, alphas_forward, alphas_backwards, vids], class_report, class_report_with_dict
    
    '''
    将分类识别结果可视化 
    '''
    def visualize_results(self, i, predictions, true_label):
           predictions, true_label = predictions, true_label
           labels_names = self.label_names 
           plt.grid(False)
           x = range(len(labels_names))
           y = predictions 
           thisplot = plt.bar(x, y, color = "#777777")
           _ = plt.xticks(range(len(labels_names)),labels_names) 
           plt.ylim([0, 1])
           predicted_label = np.argmax(predictions)
           true_label = np.argmax(true_label)
           thisplot[predicted_label].set_color('red')
           thisplot[true_label].set_color('blue')
           for a,b in zip(x, y):
               plt.text(a, b + 0.05, '{:2.2f}%'.format(b * 100), ha = 'center', va = 'bottom', fontsize = 7)
        
        
    '''
    可视化预测结果对应的emoji表情 
    '''
    def plot_image(self, i, predictions, true_label, image):
            predictions, true_label, image = predictions, true_label, image
            labels_names = self.label_names
            plt.grid(False) 
            plt.xticks([])
            plt.yticks([])
            
            plt.imshow(image)
            predicted_label = np.argmax(predictions)
            true_label = np.argmax(true_label)
            if predicted_label == true_label:
                color = 'blue' # 预测正确的话用蓝色表示
            else:
                color = 'red'  # 预测错误的话用红色表示
            plt.xlabel("Predicted: {}, True: {}".format(labels_names[predicted_label], labels_names[true_label]), color = color)
    
    '''
    评估模型 
    ''' 
    def evaluate(self):
        '''
        显示训练过程中loss和val_loss的变化情况 
        '''
        def show_train_history(y1, y2, epoch = 100, loss = False):
            x = [i + 1 for i in range(0, epoch)]
            y1 = [y for y in y1]
            y2 = [y for y in y2]
            plt.plot(x, y1)
            plt.plot(x, y2)
            plt.title('Train History')
            plt.xlabel('Epoch') 
            if loss:
                plt.ylabel('loss')
            else:
                plt.ylabel('acc(%)')
            plt.legend(['train', 'validation'], loc = 'upper left') 
            if loss:
                plt.savefig('{}_{}_{}_loss_history_test.png'.format(self.model_type, self.classification_mode, self.modality))
            else:
                plt.savefig('{}_{}_{}_acc_history_test.png'.format(self.model_type, self.classification_mode, self.modality))
            plt.show()
            plt.close()
                 

        '''
        画出混淆矩阵 
        '''
        def plot_confusion_matrix(cm, classes, normalize = True, title = 'Confusion matrix', cmap = plt.cm.Blues):
            if normalize == True:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
            plt.title(title, fontsize = 14)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation = 45)
            plt.yticks(tick_marks, classes)
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment = "center", color = "white" if cm[i, j] > thresh else "black")
            
            plt.ylabel('True label', fontsize = 14)
            plt.xlabel('Predicted label', fontsize = 14)
            plt.savefig('{}_{}_{}_confusion_matrix.png'.format(self.model_type, self.classification_mode, self.modality))
        
        
        '''
        返回一个dict，key是dialogue ID，value是该ID包含的utterance的数量
        '''
        def get_dialogs_utterances():
            data = np.load('{}_data.npy'.format(self.classification_mode), allow_pickle = True)
            dialogs_utterances = dict()
            for i in range(len(data)):
                if data[i]['split'] == 'test':
                    dialogue_id = int(data[i]['dialog'])
                    if dialogue_id not in dialogs_utterances:
                        dialogs_utterances[dialogue_id] = 1
                    else:
                        dialogs_utterances[dialogue_id] = dialogs_utterances[dialogue_id] + 1
            return dialogs_utterances
        
        predictions, test_y = np.load('{}_{}_{}_prediction.npy'.format(self.model_type, self.classification_mode, self.modality), allow_pickle = True)
        print('情感分析任务: {} 模态: {}'.format(self.classification_mode, self.modality))
        
        best_fscore, best_loss, best_label, best_predict, best_mask, test_class_report, train_losses, train_accuracies, valid_losses, valid_accuracies, epoch = np.load('{}_{}_{}.npy'.format(self.model_type, self.classification_mode, self.modality), allow_pickle = True)
        print('在测试集上的性能: F1-score: {:5.3f}, accuracy: {}%'.format(best_fscore, round(accuracy_score(best_label, best_predict, sample_weight = best_mask) * 100, 2)))
         
        print('训练过程中loss的变化情况:')
        show_train_history(train_losses, valid_losses, epoch = epoch, loss = True)
        print('训练过程中accuracy的变化情况: ')
        show_train_history(train_accuracies, valid_accuracies, epoch = epoch, loss = False)
        print('分类报告:\n')  
        print(classification_report(best_label, best_predict,  sample_weight = best_mask, target_names = self.label_names, digits = 3))
        
        print('混淆矩阵:\n') 
        cm = confusion_matrix(best_label, best_predict, sample_weight = best_mask)
        plot_confusion_matrix(cm, self.label_names, True) 
        
        predictions, test_y = np.load('{}_{}_{}_prediction.npy'.format(self.model_type, self.classification_mode, self.modality), allow_pickle = True)
    
        print('可视化识别结果: ')  
        dialogs_utterances = get_dialogs_utterances()  
        dialogue_ID = self.dialogueID
        num_images = dialogs_utterances[dialogue_ID]
     
        plt.figure(figsize = (18, 12))  
        num_rows, num_cols = math.ceil((num_images * 2) / 4), 4
        for i in range(num_images):
            plt.subplot(num_rows, num_cols, 2 * i + 1)
            plt.title('utterance {}'.format(i))
            image_index = np.argmax(predictions[dialogue_ID][i]) 
            image = mpimg.imread('/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/{}_emojis/{}.png'.format(self.classification_mode, self.label_names[image_index])) 
            self.plot_image(i, predictions[dialogue_ID][i], test_y[dialogue_ID][i], image)
            plt.subplot(num_rows, num_cols, 2 * i + 2)
            self.visualize_results(i, predictions[dialogue_ID][i], test_y[dialogue_ID][i])
        plt.tight_layout()
        plt.savefig('/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/results/{}_{}_dialogue_{}_result.png'.format(self.modality, self.classification_mode, dialogue_ID))


if __name__ == "__main__":
    print('多模态情感分析的Multi-party Conversational RNN方法')
    
    classification_mode = input('请输入情感分析任务：Emotion or Sentiment\n')
    while classification_mode.lower() not in ['emotion', 'sentiment']: 
        print('请输入正确的情感分析任务\n')
        classification_mode = input('请输入情感分析任务：Emotion or Sentiment\n')
        if classification_mode == 'exit':
            exit()
            
    modality = input('请输入需要处理的模态信息：text, audio, bimodal or multimodal\n')
    while modality.lower() not in ['text', 'audio', 'bimodal', 'multimodal']: 
        print('请输入正确的模态信息\n')
        modality = input('请输入需要处理的模态信息：text, audio, bimodal or multimodal\n')
        if modality == 'exit':
            exit()
    
    dialogueID = input('请输入需要识别的测试集对话编号(范围：0 - 279)：\n')
     
    model = TrainEvaluateMELD(classification_mode.lower(), modality.lower(), int(dialogueID))
    model.load_data()
    #model.train()
    model.evaluate() 
    
    

