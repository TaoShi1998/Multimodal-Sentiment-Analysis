#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:23:20 2020

@author: Tao Shi 
""" 

import numpy as np 
import itertools
from tensorflow.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D
# Masking: 使用覆盖值覆盖序列，以跳过时间步
# TimeDistributed: 对三维张量的每个时间步长应用相同的全连接操作
from tensorflow.keras.layers import Lambda, LSTM, TimeDistributed, Masking, Bidirectional, Dropout, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model
# backend: 后端模块 
import tensorflow.keras.backend as backend   
from data_processor import Dataloader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
#from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
from tensorflow.keras.utils import plot_model
 
# baseline模型 
class Baseline:
    def __init__(self, classify, modality, dialogueID):
        self.classification_mode = classify # 情绪识别或者情感识别
        self.modality = modality            # 单模态(文本或者音频)或双模态(文本+音频) 
        self.dialogueID = dialogueID
        self.PATH = "/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/data/weights/{}_weights_{}.hdf5".format(modality,self.classification_mode)
        print("Baseline模型的分类任务为: {} Recognition".format(self.classification_mode.capitalize()))
      
     
    def load_data(self): 
        print('加载数据')
        self.data = Dataloader(mode = self.classification_mode)
        
        if self.modality == "text":
            self.data.load_text_data()
        elif self.modality == "audio":
            self.data.load_audio_data()
        elif self.modality == "bimodal":
            self.data.load_bimodal_data()
        else:
            print('模态信息输入有误，请重新输入')
            exit()
        
        # 导入训练集、验证集和测试集 
        self.train_x = self.data.train_dialogue_features
        self.val_x = self.data.val_dialogue_features
        self.test_x = self.data.test_dialogue_features
        
        self.train_y = self.data.train_dialogue_label
        self.val_y = self.data.val_dialogue_label
        self.test_y = self.data.test_dialogue_label 
        
        # 导入用于排除padding的mask 
        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask 
        self.test_mask = self.data.test_mask
        
        # 导入数据集所对应的dialogue ID 和 utterance ID
        self.train_id = self.data.train_dialogue_ids.keys()
        self.val_id = self.data.val_dialogue_ids.keys()
        self.test_id = self.data.test_dialogue_ids.keys()
        
        # 导入在数据集中最长的对话所包含的语句数量 
        self.sequence_length = self.train_x.shape[1]
        self.classes = self.train_y.shape[2]
        
        self.label_index = self.data.label_index.keys()
    
        
    
    '''
    搭建基于CNN和LSTM的文本模态情感/情绪识别的模型  
    '''
    def create_text_model(self): 
        
        def slicer(x, index):
            return x[:, backend.constant(index, dtype = 'int32'), :]
        
        def slicer_output_shape(input_shape):
            shape = list(input_shape)
            assert len(shape) == 3  
            new_shape = (shape[0], shape[2])
            return new_shape
        
        def reshaper(x):
            return backend.expand_dims(x, axis = 3)
        
        def flattener(x):
            x = backend.reshape(x, [-1, x.shape[1] * x.shape[3]])
            return x
        
        def flattener_output_shape(input_shape):
            shape = list(input_shape) 
            new_shape = (shape[0], 3 * shape[3])
            return new_shape
        
        def stack(x): 
            return backend.stack(x, axis = 1)
        
        self.vocabulary_size = self.data.W.shape[0]  # input_dimension
        self.embedding_dim = self.data.W.shape[1]    # output_dimension 
        self.filter_sizes = [3, 4, 5]
        self.num_filters = 512 
          
        print("创建文本模态的情感识别模型")
         
        sentence_length = self.train_x.shape[2]
        
        embedding = Embedding(input_dim = self.vocabulary_size, output_dim = self.embedding_dim, weights = [self.data.W], 
                              input_length = sentence_length, trainable = False)
        conv_0 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[0], self.embedding_dim), 
                        padding = 'valid', kernel_initializer = 'normal', activation = 'relu')
        conv_1 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[1], self.embedding_dim), 
                        padding = 'valid', kernel_initializer = 'normal', activation = 'relu')
        conv_2 = Conv2D(self.num_filters, kernel_size=(self.filter_sizes[2], self.embedding_dim), 
                        padding = 'valid', kernel_initializer = 'normal', activation = 'relu')
        maxpool_0 = MaxPool2D(pool_size = (sentence_length - self.filter_sizes[0] + 1, 1), strides=(1, 1), padding = 'valid')
        maxpool_1 = MaxPool2D(pool_size = (sentence_length - self.filter_sizes[1] + 1, 1), strides=(1, 1), padding = 'valid')
        maxpool_2 = MaxPool2D(pool_size = (sentence_length - self.filter_sizes[2] + 1, 1), strides=(1, 1), padding = 'valid')
        dense_func = Dense(100, activation = 'tanh', name = "dense")
         
        inputs = Input(shape = (self.sequence_length, sentence_length), dtype = 'int32')   # 输入层
        cnn_output = []  
        local_input = Lambda(slicer, output_shape = slicer_output_shape, arguments = {"index": 0})(inputs)   # Lambda层实现数据切片 
        emb_output = embedding(local_input)      # Embedding层
        reshape = Lambda(reshaper)(emb_output)   # Lambda层实现维度扩充 
        concatenated_tensor = Concatenate(axis = 1)([maxpool_0(conv_0(reshape)), maxpool_1(conv_1(reshape)), maxpool_2(conv_2(reshape))])  # 拼接层
        flatten = Lambda(flattener, output_shape = flattener_output_shape,)(concatenated_tensor)   # Lambda层实现一维化处理 
        dense_output = dense_func(flatten)       # 全连接层 
        dropout = Dropout(0.5)(dense_output)     # Dropout层 
        cnn_output.append(dropout)
    
        cnn_outputs = Lambda(stack)(cnn_output)       # Lambda层实现在第一维上将秩为0的张量列表堆叠成秩为1的张量
        masked = Masking(mask_value = 0)(cnn_outputs) # Mask层使用覆盖值0覆盖输入序列 
        lstm = Bidirectional(LSTM(512, activation = 'relu', return_sequences = True, dropout = 0.4))(masked) # Bidirectional LSTM层
        lstm = Bidirectional(LSTM(512, activation = 'relu', return_sequences = True, dropout = 0.4), name = "utter")(lstm)  # Bidirectional LSTM层
        output = TimeDistributed(Dense(self.classes, activation = 'softmax'))(lstm) # TimeDistributed层 

        model = Model(inputs, output) # TimeDistributed的输入与输出均是三维的 
        
        plot_model(model, to_file = 'baseline_text_model.png', show_shapes = True, show_layer_names = False) # 画出模型结构 
         
        return model
      
     
    '''
    搭建基于Bi-Directional RNN的音频模态情感/情绪识别的模型 
    '''
    def create_audio_model(self):
        self.embedding_dim = self.train_x.shape[2]
         
        print("创建音频模态的情感识别模型")
        
        inputs = Input(shape = (self.sequence_length, self.embedding_dim), dtype = 'float32')                    # 输入层
        masked = Masking(mask_value = 0)(inputs)                                                                 # Mask层
        lstm = Bidirectional(LSTM(512, activation = 'relu', return_sequences = True, dropout = 0.4))(masked)     # Bidirectional LSTM层
        output = TimeDistributed(Dense(self.classes, activation = 'softmax'))(lstm)                              # 输出层
        
        model = Model(inputs, output)
        
        plot_model(model, to_file = 'baseline_audio_model.png', show_shapes = True, show_layer_names = False)   # 画出模型结构 
        
        return model 
       
    
    '''
    搭建基于LSTM的文本+音频双模态情感/情绪识别的模型  
    '''
    def create_bimodal_model(self):
        self.embedding_dim = self.train_x.shape[2] 
        
        print("创建文本+音频双模态的情感识别模型")
        
        inputs = Input(shape = (self.sequence_length, self.embedding_dim), dtype = 'float32')                     # 输入层
        masked = Masking(mask_value = 0)(inputs)                                                                  # Mask层
        lstm = Bidirectional(LSTM(512, activation = 'relu', return_sequences = True, dropout = 0.4))(masked)      # Bidirectional LSTM层
        lstm = Bidirectional(LSTM(512, activation = 'relu', return_sequences = True, dropout = 0.4))(lstm)        # Bidirectional LSTM层
        output = TimeDistributed(Dense(self.classes, activation = 'softmax'))(lstm)
        
        model = Model(inputs, output)
        
        plot_model(model, to_file = 'baseline_bimodal_model.png', show_shapes = True, show_layer_names = False)  # 画出模型结构 
        
        return model
      
        
    ''' 
    训练模型 
    '''
    def train_model(self):
        # 设置超参数
        self.epochs = 100
        self.batch_size = 32
        
        if self.modality == "audio":
            model = self.create_audio_model()
            model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', sample_weight_mode='temporal')
        elif self.modality == "text":
            model = self.create_text_model()
            model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy',  sample_weight_mode='temporal')
        elif self.modality == "bimodal":
            model = self.create_bimodal_model()
            model.compile(optimizer = 'adam', loss='categorical_crossentropy', sample_weight_mode='temporal')
         
        #model.summary()
        
        checkpoint = ModelCheckpoint(self.PATH, monitor = 'val_loss', verbose = 1, save_best_only = True, mode = 'auto')
        early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10)                                       # 防止过拟合 
        reduce_learning_rate = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 5, verbose = 1)  # val_loss不提升的时候降低学习率 
          
        history = model.fit(self.train_x, self.train_y,
                            epochs = self.epochs,
                            batch_size = self.batch_size, 
                            shuffle = True, 
                            callbacks=[early_stopping, checkpoint, reduce_learning_rate],
                            sample_weight = self.train_mask,
                            validation_data = (self.val_x, self.val_y, self.val_mask))
            
        
        np.save('{}_{}_history.npy'.format(self.modality, self.classification_mode), history.history) # 保存history
        model.save('baseline_{}_{}_modal.h5'.format(self.modality, self.classification_mode)) 
    
    
    '''  
    显示训练过程中loss和val_loss的变化情况以及accuracy和val_accuracy的变化情况
    '''
    def plot_train_history(self, loss = False):
        
        def show_train_history(history):
            if loss:
                x1 = history['loss']
                x2 = history['val_loss']
            else:
                x1 = history['acc']
                x2 = history['val_acc']
                # 将小数化为百分数 
                x1 = [x * 100 for x in x1]
                x2 = [x * 100 for x in x2]
            plt.plot(x1)
            plt.plot(x2)
            plt.title('Train History')
            plt.xlabel('Epoch') 
            if loss:
                plt.ylabel('loss')
            else:
                plt.ylabel('acc(%)')
            plt.legend(['train', 'validation'], loc = 'upper left') 
        
        history = np.load('{}_{}_history.npy'.format(self.modality, self.classification_mode), allow_pickle = True)
        history = history.tolist()
        plt.figure(figsize = (6, 4))
        show_train_history(history)
        if loss:
            plt.savefig('{}_{}_train_loss.png'.format(self.modality, self.classification_mode))
        else:
            plt.savefig('{}_{}_train_accuracy.png'.format(self.modality, self.classification_mode))
    
 
    '''
    画出混淆矩阵 
    '''
    def plot_confusion_matrix(self, cm, classes, normalize = True, title = 'Confusion matrix', cmap = plt.cm.Blues):
            if normalize == True:  # 归一化处理
                cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]
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
            plt.savefig('{}_{}_confusion_matrix.png'.format(self.modality, self.classification_mode))
         
            
    '''
    将分类识别结果可视化 
    '''
    def visualize_results(self, i, predictions, true_label):
           predictions, true_label = predictions, true_label
           labels_names = self.label_index 
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
        labels_names = list(self.label_index)
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
    对训练好的模型进行评估并将识别结果可视化 
    ''' 
    def evaluate_model(self):
             
        '''
        返回一个dict，key是Dialogue_ID，value是该ID包含的utterance的数量
        '''
        def get_dialogs_utterances():
            data = self.data.data
            dialogs_utterances = dict()
            for i in range(len(data)):
                if data[i]['split'] == 'test':
                    dialogue_id = int(data[i]['dialog'])
                    if dialogue_id not in dialogs_utterances:
                        dialogs_utterances[dialogue_id] = 1
                    else:
                        dialogs_utterances[dialogue_id] = dialogs_utterances[dialogue_id] + 1
            return dialogs_utterances
        

        model = load_model('baseline_{}_{}_modal.h5'.format(self.modality, self.classification_mode))

        test_x = self.test_x
        test_y = self.test_y
        test_mask = self.test_mask 
         
        predictions = model.predict(test_x)
        
        true_label=[]      # 真实标签
        predicted_label=[] # 预测标签  
          
        for i in range(test_x.shape[0]):  
            for j in range(predictions.shape[1]):
                if test_mask[i, j] == 1:
                    true_label.append(np.argmax(test_y[i, j]))
                    predicted_label.append(np.argmax(predictions[i, j]))
        
        cm = confusion_matrix(true_label, predicted_label) 
        labels = list(self.label_index)
         
        print("分类报告:\n")  
        print(classification_report(true_label, predicted_label, target_names = labels, digits = 3))
        
        print('混淆矩阵:\n')
        self.plot_confusion_matrix(cm, labels, True)
         
        print('可视化识别结果: ')  
        dialogs_utterances = get_dialogs_utterances()  
        dialogue_ID = self.dialogueID
        num_images = dialogs_utterances[dialogue_ID]
        print(predictions[dialogue_ID][9])
        plt.figure(figsize = (18, 12)) 
        num_rows, num_cols = math.ceil((num_images * 2) / 4), 4
        for i in range(num_images):
            plt.subplot(num_rows, num_cols, 2 * i + 1)
            plt.title('utterance {}'.format(i))
            image_index = np.argmax(predictions[dialogue_ID][i]) 
            image = mpimg.imread('/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/{}_emojis/{}.png'.format(self.classification_mode, labels[image_index])) 
            self.plot_image(i, predictions[dialogue_ID][i], test_y[dialogue_ID][i], image)
            plt.subplot(num_rows, num_cols, 2 * i + 2)
            self.visualize_results(i, predictions[dialogue_ID][i], test_y[dialogue_ID][i])
        plt.tight_layout()
        plt.savefig('/Users/apple/Downloads/2020毕业设计/面向对话视频的情感分类任务/Multimodal_Sentiment_Analysis/results/{}_{}_dialogue_{}_result.png'.format(self.modality, self.classification_mode, dialogue_ID))
        
                  
if __name__ == "__main__":
    print('多模态情感分析的Baseline方法')
    
    classify = input('请输入情感分析任务：Emotion or Sentiment\n')
    while classify.lower() not in ['emotion', 'sentiment']: 
        print('请输入正确的情感分析任务\n')
        classify = input('请输入情感分析任务：Emotion or Sentiment\n')
        if classify == 'exit':
            exit()
    modality = input('请输入需要处理的模态信息：text, audio or bimodal\n')
    while modality.lower() not in ['text', 'audio', 'bimodal']: 
        print('请输入正确的模态信息\n')
        modality = input('请输入需要处理的模态信息：text, audio or bimodal\n')
        if modality == 'exit':
            exit()
    dialogueID = input('请输入需要识别的测试集对话编号(范围：0 - 279)：\n')
    
    model = Baseline(classify.lower(), modality.lower(), int(dialogueID))
    model.load_data() 
    #model.create_text_model()
    #model.create_audio_model()
    #model.create_bimodal_model()
    #model.train_model()
    #model.plot_train_history(False)
    #model.plot_train_history(False)
    model.evaluate_model() 
    


        
