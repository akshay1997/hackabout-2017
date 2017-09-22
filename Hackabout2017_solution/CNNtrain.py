"""
Python 2.7
"""
import numpy as np
seed=1337
np.random.seed(seed)  # for reproducibility

import cPickle as pkl
import gzip
import keras
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Merge, LSTM
import h5py
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, MaxPooling1D, GlobalMaxPooling1D
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from keras.layers.convolutional import Conv1D
from keras.models import model_from_json
from process import opfile,get_accuracy

batch_size = 64
nb_filter = 100
filter_length = 3
hidden_dims = 100
nb_epoch = 100
position_dims = 50

print "Load dataset"
f = gzip.open('embeddings/sem-relations.pkl.gz', 'rb')
yTrain, sentenceTrain, positionTrain1, positionTrain2 = pkl.load(f)	
yTest, sentenceTest, positionTest1, positionTest2  = pkl.load(f)
f.close()

max_position = max(np.max(positionTrain1), np.max(positionTrain2))+1

n_out = max(yTrain)+1
train_y_cat = np_utils.to_categorical(yTrain, n_out)
m_out = max(yTest)+1
test_y_cat = np_utils.to_categorical(yTest, m_out)

print "sentenceTrain: ", sentenceTrain.shape
print "positionTrain1: ", positionTrain1.shape
print "yTrain: ", yTrain.shape




print "sentenceTest: ", sentenceTest.shape
print "positionTest1: ", positionTest1.shape
print "positionTest2: ", positionTest2.shape
#print "yTest: ", yTest.shape


f = gzip.open('embeddings/embeddings.pkl.gz', 'rb')
embeddings = pkl.load(f)
f.close()

print "Embeddings: ",embeddings.shape

distanceModel1 = Sequential()
distanceModel1.add(Embedding(max_position, position_dims, input_length=positionTrain1.shape[1]))

distanceModel2 = Sequential()
distanceModel2.add(Embedding(max_position, position_dims, input_length=positionTrain2.shape[1]))

wordModel = Sequential()
wordModel.add(Embedding(embeddings.shape[0], embeddings.shape[1], input_length=sentenceTrain.shape[1], weights=[embeddings], trainable=False))


model = Sequential()
model.add(Merge([wordModel, distanceModel1, distanceModel2], mode='concat'))
model.add(Conv1D(activation="tanh", padding="same", strides=1, filters=100, kernel_size=3))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.20))
model.add(Dense(n_out, activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='Adam', metrics=['accuracy'])
model.summary()

print "Start training"
filepath="model_files/hack.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')
callbacks_list = [checkpoint]      
model.fit([sentenceTrain, positionTrain1, positionTrain2], train_y_cat, batch_size=batch_size,validation_split=0.1, verbose=1, epochs=120, callbacks=callbacks_list)
model.save_weights("model_files/hack.best.hdf5")
print("Saved weights to disk")

print ("Done Training")

