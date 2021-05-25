import os, glob
import csv
import random
import numpy as np

path_to_train_2d_datasets = os.path.join('..', '001 csv','outputs','train','*.csv')
train_2d_files = glob.glob(path_to_train_2d_datasets)
train = np.empty((len(train_2d_files),28,28))
train_label = np.empty(len(train_2d_files))

path_to_test_2d_datasets = os.path.join('..', '001 csv','outputs','test','*.csv')
test_2d_files = glob.glob(path_to_test_2d_datasets)
test = np.empty((len(test_2d_files),28,28))
test_label = np.empty(len(test_2d_files))

path_to_save_train = os.path.join("train.npz")
path_to_save_validation = os.path.join("validation.npz")
path_to_save_test = os.path.join("test.npz")


for data_idx, data_path in enumerate(train_2d_files):
    repalced_data_path = data_path.replace('../001 csv/outputs/train/#','')
    label = repalced_data_path[25]
    train_label[data_idx] = label
    csv_data = csv.reader(open(data_path))
    temp = []
    for row in csv_data:
        temp.append(row)
    for i in range(28):
        for j in range(28):
            train[data_idx,i,j] = temp[i][j]


for data_idx, data_path in enumerate(test_2d_files):
    repalced_data_path = data_path.replace('../001 csv/outputs/test/#','')
    label = repalced_data_path[24]
    test_label[data_idx] = label
    csv_data = csv.reader(open(data_path))
    temp = []
    for row in csv_data:
        temp.append(row)
    for i in range(28):
        for j in range(28):
            test[data_idx,i,j] = temp[i][j]


concat_data = np.concatenate((train, test), axis=0)
concat_label = np.concatenate((train_label,test_label), axis=0)
print(concat_data.shape)
print(concat_label.shape)

shuffle = np.arange(concat_data.shape[0])
np.random.shuffle(shuffle)

data = concat_data[shuffle]
labels = concat_label[shuffle]

split1 = int(0.7 * data.shape[0])
split2 = int(0.9 * data.shape[0])

[train_set, validation_set, test_set] = np.split(data, [split1,split2])
[train_labels, validation_labels, test_labels] = np.split(labels, [split1,split2])

np.savez(path_to_save_train, data=train_set, label=train_labels)
np.savez(path_to_save_validation, data=validation_set, label=validation_labels)
np.savez(path_to_save_test, data=test_set, label=test_labels)