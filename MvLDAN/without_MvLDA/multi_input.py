
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 10:29:41 2019
@author: dhanunjaya
"""


import numpy as np
from keras import layers
from keras import Input
from keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from keras.utils import to_categorical

x_train = np.load('data/DCASE2017/x_train_mean_bg.npy')
y_train = to_categorical(np.load('data/DCASE2017/y_train_mean_bg.npy'))

x_val = np.load('data/DCASE2017/x_val_mean_bg.npy')
y_val = to_categorical(np.load('data/DCASE2017/y_val_mean_bg.npy'))

x_test = np.load('data/DCASE2017/x_eval_mean_bg.npy')
y_test = to_categorical(np.load('data/DCASE2017/y_test_mean_bg.npy'))

bg_input = Input(shape=(6144,), name='background')
bg = layers.Dense(512, activation='relu')(bg_input)
bg = layers.Dense(256, activation='relu')(bg)
bg_out = Model(inputs=bg_input, outputs=bg)

fg_input = Input(shape=(6144,), name='foreground')
fg = layers.Dense(512, activation='relu')(fg_input)
fg = layers.Dense(256, activation='relu')(fg)
fg_out = Model(inputs=fg_input, outputs=fg)

combined = layers.concatenate([bg, fg])
x = layers.Dense(256, activation="relu")(combined)
out = layers.Dense(15, activation='softmax')(x)

model = Model(inputs=[bg_input, fg_input], outputs=out)


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=["accuracy"])
print(model.summary())

history = model.fit([x_train[0],x_train[1]], y_train[0], 
                    epochs=20, 
                    batch_size=32,
                    verbose=1, 
                    validation_data=([x_val[0],x_val[1]], y_val[0]))

results = model.evaluate([x_test[0],x_test[1]], y_test[0])

model.save('multi_input.h5')

print(results)
