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

np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)
cvscores = []
for i in range(2):
	#gnb = GaussianNB()
	#y_pred = gnb.fit(x_train[i], y_train[i]).predict(x_test[i])
	neigh = KNeighborsClassifier(n_neighbors=5, metric='cosine')
	neigh.fit(x_train[i], y_train[i])
	y_pred = neigh.predict(x_test[i])
	#svm_clas1 = svm.SVC(kernel='linear', C=0.01)
	#svm_clas1.fit(x_train[i], y_train[i]) 
	#y_pred = svm_clas1.predict(x_test[i])
	cvscores.append(metrics.accuracy_score(y_test[i], y_pred)*100)
	print("Accuracy: ",metrics.accuracy_score(y_test[i], y_pred)*100,"%")
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
