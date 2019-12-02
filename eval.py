# coding=utf-8

import numpy as np
import pandas as pd
import tensorflow as tf

import keras 
import keras.backend as K
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation,Dropout,BatchNormalization,Layer
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

import pretty_midi
import warnings
import os
import h5py
import pandas as pd
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()

config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config)) # 此处不同

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)
    
    def build(self, input_shape):
        assert len(input_shape)==3
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight', 
                                 shape=(input_shape[1], input_shape[1]),
                                 initializer='uniform',
                                 trainable=True)

        self.b = self.add_weight(name='att_bias',
                                 shape=(input_shape[1],),
                                 initializer='uniform',
                                 trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        # inputs.shape = (batch_size, time_steps, seq_len)
        x = K.permute_dimensions(inputs, (0, 2, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        # general
        
        a = K.softmax(K.tanh(K.dot(x, self.W) + self.b))
        
        #a = K.softmax((K.dot(x, self.W)))
        a = K.permute_dimensions(a, (0, 2, 1))
        outputs = a * inputs
        outputs = K.sum(outputs, axis=1)
        return outputs


if __name__ == "__main__":
    # 加载流派类别 
    genre_df = pd.read_csv("prepared/genres.csv")
    label_dict = {row.Genre: idx for idx, row in genre_df.iterrows()}
    print(label_dict)

    genres = [row.Genre for _, row in genre_df.iterrows()]
    num_classes = len(label_dict)
    
    # 加载测试集
    features_file = "prepared/labeled_features.h5"
    #features_file = "prepared/90_feature.h5"
    f = h5py.File(features_file, "r")
    features = f["features"].value
    labels = f["labels"].value

    valid_idx_df = pd.read_csv("valid_idx.csv")
    
    valid_idx = valid_idx_df.values.reshape(-1)
#    print(valid_idx.shape)

    valid_features  = features[valid_idx]
    valid_labels = labels[valid_idx]
    print(valid_features.shape)
    print(valid_labels.shape)
    input_shape = valid_features[0].shape

    valid_lbl_hot = keras.utils.to_categorical(valid_labels, num_classes)
    

    # 读取模型
    model_path = "model_save/weights-improve-0.91.hdf5"
    
    #model_path = "model_save/best.hdf5"
    model = load_model(model_path, custom_objects={'AttentionLayer': AttentionLayer})
    model.summary()
    #exit()


    # 测试模型
    #score, acc = model.evaluate(valid_features, valid_lbl_hot, batch_size=128)
    #print('Test loss:', score)
    #print('Test accuracy:', acc)

    prediction = model.predict(valid_features)
    predict_labels = np.argmax(prediction, axis=1)
    
    # 求混淆矩阵
    matrix = pd.crosstab(valid_labels, predict_labels, rownames=['label'],colnames=['predict'])

    # 每一类的准确率
    for i, row in matrix.iterrows():
        print(genres[i], ":", row.iloc[i] / np.sum(row), ",", np.sum(row))

    # matrix.columns = genres
    # matrix.index = genres

    print(matrix)

    # 求topk 准确率
    k = 1
    predict_topk = np.argpartition(prediction, num_classes - k)[:, -k:]
    #print(predict_topk)

    assert valid_labels.shape[0] == predict_topk.shape[0]
    num_valid = valid_labels.shape[0]
    n_acc = 0

    type_num_cnt = np.array([0 for _ in range(num_classes)])
    type_acc_cnt = np.array([0 for _ in range(num_classes)])

    for i in range(num_valid):
        type_num_cnt[valid_labels[i]] += 1
        if valid_labels[i] in predict_topk[i]:
            n_acc += 1
            type_acc_cnt[valid_labels[i]] += 1

    acc_topK = n_acc*1.0/num_valid
    type_acc_topk = type_acc_cnt / type_num_cnt

    print("top %d: %f" % (k, acc_topK))
    print(type_acc_topk)

    print("统计：precision recall f1")
    n = len(matrix)
    P = np.zeros(n, float)
    R = np.zeros(n, float)
    for i in range(len(matrix[0])):
        row_sum, col_sum = sum(matrix[i]), sum(matrix[r][i] for r in range(n))
        P[i] = (matrix[i][i] / float(row_sum))
        R[i] = (matrix[i][i] / float(col_sum))
        print("p%s: %s,r%s: %s" % (i, P[i], i, R[i]))

    macro_p = np.average(P)
    macro_r = np.average(R)
    macro_f1 = 2 * macro_p * macro_r / (macro_p + macro_r)
    print("Acc: %s, Macro-P: %s, Macro-R: %s, Macro-F1: %s" % (acc_topK, macro_p, macro_r, macro_f1))
