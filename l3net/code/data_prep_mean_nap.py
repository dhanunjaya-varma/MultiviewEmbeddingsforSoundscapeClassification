import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics
from scipy.spatial import distance
from scipy.special import softmax
from sklearn import svm
from sklearn.model_selection import train_test_split


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
q = 1
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
x_train_new_bg = np.transpose(np.transpose(x_train) - np.dot(mat,np.transpose(x_train)))
x_test_new_bg = np.transpose(np.transpose(x_test) - np.dot(mat,np.transpose(x_test)))

p = 0
q = 150
x_train = np.load('../data/x_audio_train.npy')
y_train = np.load('../data/y_audio_train.npy')
x_test = np.load('../data/x_audio_eval.npy')
y_test = np.load('../data/y_audio_eval.npy')
#basis = np.load('../basis/basis_full_bg.npy')

x_train_bg = np.load('../data/x_foreground_train.npy')
y_train_bg = np.load('../data/y_foreground_train.npy')
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
x_train_new_fg = np.transpose(np.transpose(x_train) - np.dot(mat,np.transpose(x_train)))
x_test_new_fg = np.transpose(np.transpose(x_test) - np.dot(mat,np.transpose(x_test)))

x_train = []
x_test = []

x_train.append(x_train_new_bg)
x_train.append(x_train_new_fg)
x_test.append(x_test_new_bg)
x_test.append(x_test_new_fg)

x_train = np.array(x_train)
x_test = np.array(x_test)

x_train = np.swapaxes(x_train, 0, 1)
x_test = np.swapaxes(x_test, 0, 1)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=3)

x_train = np.swapaxes(x_train, 0, 1)
x_val = np.swapaxes(x_val, 0, 1)
x_test = np.swapaxes(x_test, 0, 1)
print(x_train.shape)
print(x_test.shape)

y_train_new = []
for i in range(y_train.shape[0]):
    y_train_new.append(np.repeat(y_train[i], 2))

y_test_new = []
for i in range(y_test.shape[0]):
    y_test_new.append(np.repeat(y_test[i], 2))

y_val_new = []
for i in range(y_val.shape[0]):
    y_val_new.append(np.repeat(y_val[i], 2))

y_train_new = np.array(y_train_new)
y_test_new = np.array(y_test_new)
y_val_new = np.array(y_val_new)
y_train_new = np.swapaxes(y_train_new, 0, 1)
y_test_new = np.swapaxes(y_test_new, 0, 1)
y_val_new = np.swapaxes(y_val_new, 0, 1)

np.save('../../MvLDAN/data/DCASE2017/x_train_mean_bg.npy', x_train)
np.save('../../MvLDAN/data/DCASE2017/x_test_mean_bg.npy', x_test)
np.save('../../MvLDAN/data/DCASE2017/x_val_mean_bg.npy', x_val)
np.save('../../MvLDAN/data/DCASE2017/y_train_mean_bg.npy', y_train_new)
np.save('../../MvLDAN/data/DCASE2017/y_test_mean_bg.npy', y_test_new)
np.save('../../MvLDAN/data/DCASE2017/y_val_mean_bg.npy', y_val_new)
