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
y_train = to_categorical(np.load('output/pairwise/y_train_mean.npy'))

x_val = np.load('output/pairwise/x_val_mean.npy')
y_val = to_categorical(np.load('output/pairwise/y_val_mean.npy'))

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
#print(x_test.shape)
#print(y_test.shape)

x_test = np.load('output/pairwise/x_test_mean.npy')
y_test = to_categorical(np.load('output/pairwise/y_test_mean.npy'))

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
cvscores = []

x_input = Input(shape=(14,), name='background')
x = layers.Dense(256, activation="relu")(x_input)
out = layers.Dense(15, activation='softmax', name='length')(x)

model = Model(inputs=x_input, outputs=out)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])
print(model.summary())

history = model.fit(x_train[0], y_train[0], 
                    epochs=20, 
                    batch_size=16,
                    verbose=1, 
                    validation_data=(x_val[0], y_val[0]))
scores = model.evaluate(x_test[0], y_test[0], verbose=0, batch_size=16)
cvscores.append(scores[1]*100)
print("Accuracy: ",scores[1]*100,"%")


x_input = Input(shape=(14,), name='foreground')
x = layers.Dense(256, activation="relu")(x_input)
out = layers.Dense(15, activation='softmax')(x)

model = Model(inputs=x_input, outputs=out)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])
print(model.summary())

history = model.fit(x_train[1], y_train[1], 
                    epochs=20, 
                    batch_size=16,
                    verbose=1, 
                    validation_data=(x_val[1], y_val[1]))
scores = model.evaluate(x_test[1], y_test[1], verbose=0, batch_size=16)
cvscores.append(scores[1]*100)
print("Accuracy: ",scores[1]*100,"%")
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
