import os
import numpy as np

classes=['beach', 'bus', 'cafe/restaurant', 'car', 'city_center',
         'forest_path', 'grocery_store', 'home', 'library', 'metro_station',
         'office', 'park', 'residential_area', 'train', 'tram']


input_path = '../feat/development/audio/' 

data = []
label = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        data.append(np.mean(np.load(filePath), axis=0))
        label.append(classes.index(clas))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_audio_train.npy', np.array(data))
np.save('../data/y_audio_train.npy', np.array(label))

input_path = '../feat/development/background/' 

data = []
label = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        data.append(np.mean(np.load(filePath), axis=0))
        label.append(classes.index(clas))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_background_train.npy', np.array(data))
np.save('../data/y_background_train.npy', np.array(label))

input_path = '../feat/development/foreground/' 

data = []
label = []


for clas in classes:
    files = os.listdir(input_path + clas)
    for file in files:
        filePath = input_path + clas + '/' + file
        data.append(np.mean(np.load(filePath), axis=0))
        label.append(classes.index(clas))

print(np.array(data).shape)
print(np.array(label).shape)
np.save('../data/x_foreground_train.npy', np.array(data))
np.save('../data/y_foreground_train.npy', np.array(label))
