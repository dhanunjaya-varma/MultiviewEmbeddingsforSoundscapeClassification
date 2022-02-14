import os
from os.path import abspath
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn import svm
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
#from sklearn.externals import joblib
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from keras import layers
from keras import Input
from keras.models import Model
from keras.utils import to_categorical


x_train = np.load('output/pairwise/x_train_mean.npy')
y_train = np.load('output/pairwise/y_train_mean.npy')

x_val = np.load('output/pairwise/x_val_mean.npy')
y_val = np.load('output/pairwise/y_val_mean.npy')

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)


x_test = np.load('output/pairwise/x_test_mean.npy')
y_test = np.load('output/pairwise/y_test_mean.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train = np.mean(x_train, axis=0)
y_train = to_categorical(y_train[0])
x_val = np.mean(x_val, axis=0)
y_val = to_categorical(y_val[0])
x_test = np.mean(x_test, axis=0)
y_test = to_categorical(y_test[0])

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

inp = Input(shape=(14,), name='background')
x = layers.Dense(256, activation="relu")(inp)
out = layers.Dense(15, activation='softmax', name='length')(x)

model = Model(inputs=inp, outputs=out)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])
print(model.summary())

history = model.fit(x_train, y_train, 
                    epochs=20, 
                    batch_size=16,
                    verbose=1, 
                    validation_data=(x_val, y_val))
scores = model.evaluate(x_test, y_test, verbose=0, batch_size=16)
print("Accuracy: ",scores[1]*100,"%")
