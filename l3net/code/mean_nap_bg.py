import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy.spatial import distance
from scipy.special import softmax
from sklearn import svm


def pca(data,nDim=0):
    # Centre data
    m = np.mean(data,axis=0)
    data = data - m
    
    # Covariance matrix
    C = np.cov(np.transpose(data))
    
    # Compute eigenvalues and sort into descending order
    evals,evecs = np.linalg.eig(C)
    indices = np.argsort(evals)
    indices = indices[::-1]
    evecs = evecs[:,indices]
    evals = evals[indices]
    if nDim>0:
        evecs = evecs[:,:nDim]
    # Produce the new data matrix
    evals = np.real(evals)
    evecs = np.real(evecs)
    #x = np.dot(np.transpose(evecs),np.transpose(data))
    # Compute the original data again
    #y=np.transpose(np.dot(evecs,x))+m
    return evecs

p = 0
q = 25
x_train = np.load('../data/x_audio_train.npy')
y_train = np.load('../data/y_audio_train.npy')
x_test = np.load('../data/x_audio_eval.npy')
y_test = np.load('../data/y_audio_eval.npy')

x_train_bg = np.load('../data/x_background_train.npy')
y_train_bg = np.load('../data/y_background_train.npy')
m = np.mean(x_train,axis=0)
x_train = x_train - m
x_test = x_test - m

x_train_new = []
x_test_new = []
#label_train = []
mean = []

for i in range(15):
	print(i)
	idx_label = np.where(y_train_bg == i)
	data = []

	for j in range(len(idx_label[0])):
		data.append(x_train_bg[idx_label[0][j]])
		#label_train.append(i)
	data1 = np.array(data)
	print(data1.shape)
	mean.append(np.mean(data1, axis = 0))

basis = pca(np.array(mean))
print(basis.shape)

mat = np.dot(basis[:,p:q], np.transpose(basis[:,p:q]))
x_train_new = np.transpose(np.transpose(x_train) - np.dot(mat,np.transpose(x_train)))
x_test_new = np.transpose(np.transpose(x_test) - np.dot(mat,np.transpose(x_test)))

svm_clas = svm.SVC(kernel='linear', C=0.01)
svm_clas.fit(x_train_new, y_train) 

y_pred = svm_clas.predict(x_test_new)
print("Accuracy: ",metrics.accuracy_score(y_test, y_pred)*100,"%")
