# coding=utf-8


import keras
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py
import os
from keras.models import Sequential, load_model
from keras.layers import  Masking, LSTM, GRU, Bidirectional,Flatten
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()

config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
set_session(tf.Session(config=config)) # 此处不同




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

f.close()

label_df = pd.DataFrame(labels, columns=["Label"])

train_lbl_df, valid_lbl_df= cut_subset_frac(label_df, np.arange(num_classes), frac)

train_features  = features[train_lbl_df.index]
valid_features  = features[valid_lbl_df.index]

train_labels = train_lbl_df.values
valid_labels = valid_lbl_df.values


train_lbl_hot = keras.utils.to_categorical(train_labels, num_classes=num_classes)
valid_lbl_hot = keras.utils.to_categorical(valid_labels, num_classes=num_classes)


print(train_features.shape)
print(valid_features.shape)


batch_size = 128
epochs = 200
dropout=0.1
lr = 0.001
    

weights_path = "model_save/best.hdf5"
checkpoint = ModelCheckpoint(weights_path, 
                            monitor='val_acc', 
                            verbose=1,
                            save_best_only=True,
                            mode='max')
callbacks_list = [checkpoint]


sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


opt = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
 
model = Sequential()
model.add(Masking(mask_value=0, batch_input_shape=(None, 50, 10)))
#model.add(Bidirectional(GRU(64,dropout_W=dropout, dropout_U=dropout, return_sequences=True), merge_mode='concat'))

model.add(Bidirectional(GRU(64, return_sequences=False), merge_mode='concat'))
#model.add(GRU(128,dropout_W=dropout, dropout_U=dropout, return_sequences=False))
#model.add(GRU(64, return_sequences=False))
#model.add(Dropout(dropout))
#model.add(Dense(64, activation="tanh"))

model.add(Dense(num_classes, activation="softmax"))


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
