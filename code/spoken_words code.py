# -*- coding: utf-8 -*-
from numpy import array
from numpy import argmax
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.layers import GlobalMaxPool1D

import pandas as pd
import numpy as np
import os
import io

np.random.seed(1337)


def train_data(path,feat):
  train_set = pd.DataFrame({"path":path,"features":feat})
  train_pd = pd.merge(train_csv,
                    train_set[["path","features"]],
                    left_on = ["path"],
                    right_on = ["path"],
                    how = "left")
  train_pd.loc[train_pd['features'] == " "]
  x_train_1 = train_pd.iloc[:,2:]
  x_train_1 = np.array(x_train_1)
  y_train_1 = train_pd.iloc[:,1:2]
  y_train_1 = np.array(y_train_1)
  return x_train_1 , y_train_1

def test_data(path,feat):
  test_set = pd.DataFrame({"path":path,"features":feat})
  test_pd = pd.merge(test_csv,
                    test_set[["path","features"]],
                    left_on = ["path"],
                    right_on = ["path"],
                    how = "left")
  test_pd.loc[test_pd['features'] == " "]
  x_test_1 = test_pd.iloc[:,1:]
  x_test_1 = np.array(x_test_1)
  return x_test_1

"""# Padding"""
def padding(x_values):
  x_list = []
  x_list_2 = []
  for length in x_values:
    zero_pad = pad_sequences(length,
                             maxlen=99,
                             dtype='float32',
                             padding = "post")
    x_list.append(zero_pad)
  for a in x_list:
    for b in a:
      x_list_2.append(b)
  x_list_2 = np.asarray(x_list_2)
  pad_post_zero = x_list_2
  return(pad_post_zero)



feat = np.load("/content/drive/My Drive/MLC/feat.npy", allow_pickle=True)
path = np.load("/content/drive/My Drive/MLC/path.npy", allow_pickle=True)
train_csv = pd.read_csv('/content/drive/My Drive/MLC/train.csv')

x_train_1, y_train_1 = train_data(path,feat)

#Let's see the if the data is balanced
np.unique(y_train_1, return_counts=True)

onehot_encode = preprocessing.OneHotEncoder()
onehot_encode.fit(y_train_1)
onehotlabels = onehot_encode.transform(y_train_1).toarray()
onehotlabels.shape

x_train_pad = padding(x_train_1)



X_train, X_test, y_train, y_test = train_test_split(x_train_pad,
                                                    onehotlabels,
                                                    stratify=onehotlabels,
                                                    test_size=0.15,
                                                    random_state=45)

"""# Model final"""


batch_size = 100
hidden_units = 950
nb_classes = 35

filepath = '/content/drive/My Drive/MLC/weights_ML.hdf5'
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_loss',
                            verbose=1,
                            save_best_only=True,
                            mode='auto')

earlystop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')

model = Sequential()
model.add(LSTM(output_dim=hidden_units,
               init='uniform',
               inner_init='uniform',
               forget_bias_init='one',
               activation='tanh',
               inner_activation='sigmoid',
               recurrent_dropout=0.2,
               input_shape=(99,13),return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

adam = keras.optimizers.Adam(lr=0.0001,beta_1 = 0.9, beta_2 = 0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics = ['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          nb_epoch=50,
          validation_data=[X_test, y_test],
          shuffle = True,
          callbacks = [checkpoint,
                       earlystop])

"""# Test data preperation"""
test_csv = pd.read_csv('/content/drive/My Drive/ML/test.csv')

x_test = test_data(path,feat)



"""# Padding test"""
x_test_pad = padding(x_test)


"""# Prediction"""
model = load_model("/content/drive/My Drive/MLC/weights_ML.hdf5")
pred =  model.predict(x_test_pad)
np.save('/content/drive/My Drive/MLC/prediction', pred)

labels = np.argmax(pred, axis=-1)

dct={0:'backward',1:'bed',2:'bird',3:'cat',4:'dog',
     5:'down',6:'eight',7:'five',8:'follow',9:'forward',
     10:'four',11:'go',12:'happy',13:'house',14:'learn',
     15:'left',16:'marvin',17:'nine',18:'no',19:'off',
     20:'on',21:'one',22:'right',23:'seven',24:'sheila',
     25:'six',26:'stop',27:'three',28:'tree',29:'two',
     30:'up',31:'visual',32:'wow',33:'yes',34:'zero'}

pred_words = [dct[k] for k in labels]
print (pred_words)

test_csv['word'] = np.array(pred_words)

test_csv_file = test_csv.to_csv("/content/drive/My Drive/MLC/result.csv", index = False)
