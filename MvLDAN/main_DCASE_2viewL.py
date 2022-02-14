#epoch_num = 50
l2 = 1e-4
learning_rate = 1e-3
od = 14

seed = 0
results = []
import config

config.l2_eig = 1.
config.lambda_cca1 = 1e-4
config.l2_eig = 1.

import numpy as np
np.random.seed(seed)
import random as rn

rn.seed(seed)
import os

os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['THEANO_FLAGS'] = "mode=FAST_RUN, device=cuda0, floatX=float32"
import scipy.io as sio
import dcase2017_2view
epoch_num = 45

config.feature_path = 'Representations/DCASE2017'
output_size = 15
batch_size = 500
result = dcase2017_2view.train_nMSAD_CNN(output_size=output_size, epoch_num=epoch_num, batch_size=batch_size, l2=l2, learning_rate=learning_rate, d=14)
print(result)
