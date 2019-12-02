# coding=utf-8

import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import os

import keras.backend as K
from keras.models import Sequential, load_model, Model
from keras.layers import  Masking, LSTM, GRU, Bidirectional,Flatten,Layer, Input, TimeDistributed
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_output(self, loss_type):

        df = pd.DataFrame({"TrainAcc": self.accuracy[loss_type], 
                           "TrainLoss": self.losses[loss_type],
                           "ValidAcc": self.val_acc[loss_type],
                           "ValidLoss": self.val_loss[loss_type]})
        df.to_csv("acc_loss.csv", index=False)



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()

config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
set_session(tf.Session(config=config)) 

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

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


# hyperparameter
frac = 0.8


def cut_subset_frac(df, labels, factor):
    assert factor > 0 and factor <= 1
    columns = df.keys()
    subset_df = pd.DataFrame(columns=columns)

    for l in labels:
        # 提取
        query_df = df.query("Label == '{}'".format(l))
        # 采样
        sample_df = query_df.sample(frac=factor)
        # 添加
        subset_df = subset_df.append(sample_df)
    res_df = df.drop(index=subset_df.index, axis=0)
    return subset_df, res_df

features_file = "prepared/labeled_features.h5"
genre_df = pd.read_csv("prepared/genres.csv")
num_classes = genre_df.shape[0]


f = h5py.File(features_file, "r")
features = f["features"].value
labels = f["labels"].value


label_df = pd.DataFrame(labels, columns=["Label"])

train_lbl_df, valid_lbl_df= cut_subset_frac(label_df, np.arange(num_classes), frac)
print(train_lbl_df.shape)
f.close()

valid_idx_df = pd.DataFrame(valid_lbl_df.index)
valid_idx_df.to_csv("check_data/d0.csv", index=False)
#exit()

train_features  = features[train_lbl_df.index]
valid_features  = features[valid_lbl_df.index]

train_labels = train_lbl_df.values
valid_labels = valid_lbl_df.values


train_lbl_hot = keras.utils.to_categorical(train_labels, num_classes=num_classes)
valid_lbl_hot = keras.utils.to_categorical(valid_labels, num_classes=num_classes)


print(train_features.shape)
print(valid_features.shape)
#exit()
input_shape = train_features[0].shape


batch_size = 64
epochs = 200
dropout=0.1
lr = 0.001
    
weights_path = "model_save/md0-{val_acc:.2f}.hdf5"
#weights_path = "model_save/best.hdf5"
checkpoint = ModelCheckpoint(weights_path, 
                            monitor='val_acc', 
                            verbose=1,
                            save_best_only=True,
                            mode='max')
history = LossHistory()

callbacks_list = [checkpoint, history]


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
 
'''
model = Sequential()
#model.add(Masking(mask_value=0, batch_input_shape=(None, 50, 10)))

#model.add(Bidirectional(GRU(64,dropout_W=dropout, dropout_U=dropout, return_sequences=True), merge_mode='concat'))

model.add(Bidirectional(GRU(64, return_sequences=True), merge_mode='concat', input_shape=(50, 10)))
#model.add(GRU(128,dropout_W=dropout, dropout_U=dropout, return_sequences=False))
#model.add(GRU(64, return_sequences=False))
#model.add(Dropout(dropout))
#model.add(Dense(64, activation="tanh"))
model.add(AttentionLayer())

model.add(Dense(num_classes, activation="softmax"))
'''

x_input = Input(shape=input_shape, dtype='float32')
#x_input = Input(shape=(50,10), dtype='float32')

l_lstm = Bidirectional(GRU(64, return_sequences=True), merge_mode='concat')(x_input)
#l_dense = TimeDistributed(Dense(256))(l_lstm)
att = AttentionLayer()(l_lstm)
preds = Dense(num_classes, activation="softmax")(att)
model = Model(x_input, preds)


model.compile(loss='categorical_crossentropy',
              optimizer= opt,
              metrics=['accuracy'])

model.summary()
#exit()


model.fit(train_features, train_lbl_hot,
        epochs=epochs,
        callbacks=callbacks_list,
        validation_data=(valid_features, valid_lbl_hot),
        )

history.loss_output("epoch")

