import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt

np.random.seed(3)

x_train = np.load('data/DCASE2017/x_train_mean_bg.npy')
y_train = np.load('data/DCASE2017/y_train_mean_bg.npy')

x_val = np.load('data/DCASE2017/x_val_mean_bg.npy')
y_val = np.load('data/DCASE2017/y_val_mean_bg.npy')

print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)
#print(x_test.shape)
#print(y_test.shape)

x_train = np.concatenate((x_train, x_val), axis=1)
y_train = np.concatenate((y_train, y_val), axis=1)

x_test = np.load('data/DCASE2017/x_eval_mean_bg.npy')
y_test = np.load('data/DCASE2017/y_test_mean_bg.npy')

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
svm_bg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_bg.fit(x_train[0], y_train[0]) 

svm_fg = svm.SVC(kernel='linear', C=0.01, probability=True)

svm_fg.fit(x_train[1], y_train[1]) 
y_pred = []

for i in range(x_test[0].shape[0]):
	pred_bg = svm_bg.predict_proba(x_test[0][i].reshape(1,-1))
	pred_fg = svm_fg.predict_proba(x_test[1][i].reshape(1,-1))
	#print(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))
	y_pred.append(np.argmax(np.mean([pred_bg, pred_fg], axis=0)))

y_pred = np.array(y_pred)
print(y_pred.shape)
print(y_test[0].shape)
print("Accuracy :", metrics.accuracy_score(y_test[0], y_pred)*100)
