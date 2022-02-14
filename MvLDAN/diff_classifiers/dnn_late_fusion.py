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
y_test = np.load('output/pairwise/y_test_mean.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
cvscores = []

x_input = Input(shape=(14,), name='background')
x = layers.Dense(256, activation="relu")(x_input)
out = layers.Dense(15, activation='sigmoid', name='length')(x)

model_bg = Model(inputs=x_input, outputs=out)


model_bg.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])
print(model_bg.summary())

history = model_bg.fit(x_train[0], y_train[0], 
                    epochs=20, 
                    batch_size=16,
                    verbose=1, 
                    validation_data=(x_val[0], y_val[0]))


x_input = Input(shape=(14,), name='foreground')
x = layers.Dense(256, activation="relu")(x_input)
out = layers.Dense(15, activation='sigmoid')(x)

model_fg = Model(inputs=x_input, outputs=out)


model_fg.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])
print(model_fg.summary())

history = model_fg.fit(x_train[1], y_train[1], 
                    epochs=20, 
                    batch_size=16,
                    verbose=1, 
                    validation_data=(x_val[1], y_val[1]))

x_test_bg = x_test[0]
x_test_fg = x_test[1]
y_test = y_test[0]

y_pred = []

for i in range(x_test_bg.shape[0]):
  pred_bg = model_bg.predict(x_test_bg[i].reshape(1,-1))
  pred_fg = model_fg.predict(x_test_fg[i].reshape(1,-1))
  #print(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))
  y_pred.append(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))

y_pred = np.array(y_pred)
print(y_pred.shape)
print(y_test.shape)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred)*100)
