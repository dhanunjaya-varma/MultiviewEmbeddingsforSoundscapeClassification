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


x_train = np.load('output/pairwise/x_train_mean.npy')
y_train = np.load('output/pairwise/y_train_mean.npy')

x_val = np.load('output/pairwise/x_val_mean.npy')
y_val = np.load('output/pairwise/y_val_mean.npy')

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
#print(x_test.shape)
#print(y_test.shape)

x_train = np.concatenate((x_train, x_val), axis=1)
y_train = np.concatenate((y_train, y_val), axis=1)

x_test = np.load('output/pairwise/x_test_mean.npy')
y_test = np.load('output/pairwise/y_test_mean.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

x_train_bg = x_train[0]
x_train_fg = x_train[1]

x_test_bg = x_test[0]
x_test_fg = x_test[1]

y_train = y_train[0]
y_test = y_test[0]

svm_bg = svm.SVC(kernel='rbf', probability=True)

svm_bg.fit(x_train_bg, y_train) 

svm_fg = svm.SVC(kernel='rbf', probability=True)

svm_fg.fit(x_train_fg, y_train) 
y_pred = []

for i in range(x_test_bg.shape[0]):
	pred_bg = svm_bg.predict_proba(x_test_bg[i].reshape(1,-1))
	pred_fg = svm_fg.predict_proba(x_test_fg[i].reshape(1,-1))
	#print(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))
	y_pred.append(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))

y_pred = np.array(y_pred)
print(y_pred.shape)
print(y_test.shape)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred)*100)
