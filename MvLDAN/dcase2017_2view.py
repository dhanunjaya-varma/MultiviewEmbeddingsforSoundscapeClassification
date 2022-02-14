import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.callbacks import Callback
from MvLDAN import th_MvLDAN_test, th_MvLDAN_test_w, th_MvLDAN_cost, th_MvLDAN, th_MvLDAN_test_test
import config
import numpy as np
import scipy.io as sio

import os
os.environ['KERAS_BACKEND'] = 'theano'
from keras.models import Sequential, Model, Input
from keras.layers import Dense, Lambda, Dropout, Conv2D, Flatten, MaxPool2D, BatchNormalization, Activation
from keras.regularizers import l2
from keras.optimizers import RMSprop, Adam, SGD
from MvLDAN import MvLDAN_gneral
from sklearn.model_selection import train_test_split

def train_model(model, epoch_num, batch_size, out_model=None, pairwise=True, d=14, MAP=-1, model_path='tmp/tmp_best2.h5'):
    str_test = [0]
    best_val_accuracy = [0]
    best_test_accuracy = [0]
    result = []
    isComputeLoss = True
    # MAP = MAP # None
    compute_all = True
    tmp_best = model_path
    best_epoch = [0]
    train_data = np.load('data/DCASE2017/x_train_mean_bg.npy').tolist()
    test_data = np.load('data/DCASE2017/x_eval_mean_bg.npy').tolist()
    valid_data = np.load('data/DCASE2017/x_val_mean_bg.npy').tolist()
    train_labels = np.load('data/DCASE2017/y_train_mean_bg.npy').tolist()
    test_labels = np.load('data/DCASE2017/y_test_mean_bg.npy').tolist()
    valid_labels = np.load('data/DCASE2017/y_val_mean_bg.npy').tolist()

    class LossHistory(Callback):
        def __init__(self, _train, _validation, _test, _batch_size=100, d=14):
            self.train_data = _train[0]
            self.train_labels = _train[1]

            self.validate_data = _validation[0]
            self.validate_labels = _validation[1]

            self.test_data = _test[0]
            self.test_labels = _test[1]

            self.batch_size = _batch_size
            self.n_view = len(self.train_data)
            self.d = d

            if out_model is None:
                self.out_model = self.model
            else:
                self.out_model = out_model

            self.history = {'tr_eigvals': [], 'val_eigvals': [], 'tr_acc': [], 'val_acc': []}

            self.test_pred = None
            self.train_pred = None

        def on_train_begin(self, logs={}):
            if isComputeLoss:
                _train = self.out_model.predict(self.train_data, self.batch_size)
                _validate = self.out_model.predict(self.validate_data, self.batch_size)
                _val_result, tr_eigvals, _, W, ms = th_MvLDAN_test(_train, self.train_labels, _validate, self.validate_labels, self.d, MAP)

                _train_resut = th_MvLDAN_test_w(W, ms, _train, self.train_labels, self.d, MAP)
                _, val_eigvals, _, _, _ = th_MvLDAN_test(_validate, self.validate_labels, _validate, self.validate_labels, self.d, MAP)
                self.history['tr_eigvals'].append(tr_eigvals)
                self.history['val_eigvals'].append(val_eigvals)
                self.history['tr_acc'].append(_train_resut)
                self.history['val_acc'].append(_val_result)
            pass
            self.on_epoch_end(-1)

        def on_batch_end(self, batch, logs={}):
            pass

        def view_result(self, _acc):
            res = ''
            if type(_acc) is not list:
                res += ((' - mean: %.4f' % (np.sum(_acc) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
                for _i in range(self.n_view):
                    for _j in range(self.n_view):
                        if _i != _j:
                            res += ('%.4f' % _acc[_i, _j]) + ','
            else:
                R = [50, 'ALL']
                for _k in range(len(_acc)):
                    res += (' R = ' + str(R[_k]) + ': ')
                    res += ((' - mean: %.4f' % (np.sum(_acc[_k]) / (self.n_view * (self.n_view - 1)))) + ' - detail:')
                    for _i in range(self.n_view):
                        for _j in range(self.n_view):
                            if _i != _j:
                                res += ('%.4f' % _acc[_k][_i, _j]) + ','
            return res

        def on_epoch_end(self, epoch, logs=None):
            _train = self.out_model.predict(self.train_data, self.batch_size)
            _validate = self.out_model.predict(self.validate_data, self.batch_size)
            _val_result, tr_eigvals, _, W, ms = th_MvLDAN_test(_train, self.train_labels, _validate, self.validate_labels, self.d, MAP)#list(range(2, 30)))

            val_eigvals_sum = np.sum(tr_eigvals[0::])
            self.str_test = ''
            if compute_all or np.sum(_val_result) > np.sum(best_val_accuracy[0]):
                best_val_accuracy[0] = _val_result
                self.train_pred = _train
                self.test_pred = self.out_model.predict(self.test_data, self.batch_size)
                _test_result, tr_eigvals, _, W, ms = th_MvLDAN_test_test(self.train_pred, self.train_labels, self.test_pred, self.test_labels, self.d, MAP)
                best_test_accuracy[0] = _test_result
                #np.save('output/pairwise/x_train.npy', np.array(self.train_pred))
                np.save('output/pairwise/y_train_mean.npy', np.array(self.train_labels))
                np.save('output/pairwise/x_validate_mean.npy', np.array(_validate))
                np.save('output/pairwise/y_val_mean.npy', np.array(self.validate_labels))
                #np.save('output/pairwise/x_test.npy', np.array(self.test_pred))
                np.save('output/pairwise/y_test_mean.npy', np.array(self.test_labels))
            print(' - val_sum: %.4f - val_results: %s %s  - val_eigenvalues: %.4f %.4f' % (val_eigvals_sum, self.view_result(_val_result), self.str_test, tr_eigvals[0], tr_eigvals[-1]))
            _val_tmp = np.concatenate(_val_result)
            result.append(np.sum(_val_result) / len(_val_tmp[_val_tmp.nonzero()]))

            if isComputeLoss:
                _train_resut = th_MvLDAN_test_w(W, ms, _train, self.train_labels, self.d, MAP)
                _, val_eigvals, _, _, _ = th_MvLDAN_test(_validate, self.validate_labels, _validate, self.validate_labels, self.d, MAP=MAP)
                self.history['tr_eigvals'].append(tr_eigvals)
                self.history['val_eigvals'].append(val_eigvals)
                self.history['tr_acc'].append(_train_resut)
                self.history['val_acc'].append(_val_result)
    print('start training...........')
    if pairwise is True:
        history = LossHistory([train_data, train_labels], [valid_data, valid_labels], [test_data, test_labels], _batch_size=batch_size, d=d)
        H = model.fit(train_data + train_labels, train_labels[0], batch_size=batch_size, epochs=epoch_num, shuffle=True, callbacks=[history], verbose=1)
        if isComputeLoss:
            import scipy.io as sio
            history.history['tr_loss'] = H.history['loss']
            sio.savemat('cnn_loss_acc_history_noisy_mnist_20.mat', history.history)
            #exit(0)
    else:
        from model import batch_generator
        model.fit_generator(batch_generator(data), steps_per_epoch=batch_size, epochs=epoch_num, validation_data=batch_generator(data, 1), validation_steps=batch_size, callbacks=[LossHistory([train_data, train_labels], [valid_data, valid_labels], [test_data, test_labels], _batch_size=batch_size, d=d)])

    tr = history.train_pred
    te = history.test_pred

    import os
    import scipy.io as sio
    ms, W, eigvals = th_MvLDAN(tr, train_labels)
    test_list = []
    train_list = []
    val_list = []
    val = np.load('output/pairwise/x_validate_mean.npy')
    for v in range(len(train_labels)):
    	test_list.append(np.dot((te[v] - ms[0][v]) / ms[1][v], W[v][:, 0:d]))
    	train_list.append(np.dot((tr[v] - ms[0][v]) / ms[1][v], W[v][:, 0:d]))
    	val_list.append(np.dot((val[v] - ms[0][v]) / ms[1][v], W[v][:, 0:d]))
    	# test_list.append(np.dot(te[v], W[v][:, 0:d]))
    np.save('output/pairwise/x_test_mean.npy', np.array(test_list))
    np.save('output/pairwise/x_train_mean.npy', np.array(train_list))
    np.save('output/pairwise/x_val_mean.npy', np.array(val_list))
    print('best_epoch:' + str(best_epoch[0]) + 'max mean accuracy:' + str(np.max(result)) + str(str_test[0]))
    return {'valid_max': best_val_accuracy[0], 'test_result': best_test_accuracy[0]}


def create_nMSAD_model(output_size, value_l2, learning_rate):
    models = []
    net_output = []
    net_input = []
    net_labels = []
    n_view = 2
    # view1 - nus_imgs

    for i in range(n_view):
        models.append(Sequential())
        models[i].add(Dense(512, input_shape=(6144,), activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
        #models[i].add(Dense(512, activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
        models[i].add(Dense(128, activation='relu', kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))
        models[i].add(Dense(output_size, kernel_regularizer=l2(value_l2), bias_regularizer=l2(value_l2)))

    for i in range(n_view):
        net_input.append(models[i].inputs[0])
        net_output.append(models[i].outputs[-1])
        net_labels.append(Input(shape=(1,)))

    loss_out = Lambda(MvLDAN_gneral, output_shape=(1,), name='ctc')(net_output + net_labels)
    model = Model(inputs=net_input + net_labels, outputs=loss_out)
    model_optimizer = Adam(lr=learning_rate, decay=0.)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=model_optimizer)
    return model, Model(inputs=net_input, outputs=net_output)

def train_nMSAD_CNN(output_size=10, epoch_num=100, batch_size=100, l2=1e-5, learning_rate=1e-3, d=14):
    # all_inx = sio.loadmat('./data/mnist_cifar10/mnist_cifar10_shuffle_inx10.mat')['mnist_cifar10_shuffle_inx10']
    result = []
    #from model import create_nMSAD_model
    times = 1
    if config.test_times == -1:
        times = 10
    for i in range(times):
        # for dd in all_data:
        if config.test_times == -1:
            inx = i
        else:
            inx = config.test_times
        model, predit_model = create_nMSAD_model(output_size, l2, learning_rate)
        model.summary()
        print("lambda_cca1: " + str(config.lambda_cca1) + '       index: ' + str(inx))
        result.append(train_model(model, epoch_num, batch_size, predit_model, d=d, model_path='tmp/dcase2017_model.h5'))
    return result
