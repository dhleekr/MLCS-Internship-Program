#==========================================#
# Title:  Data Loader
# Author: Doohyun Lee
# Date:   2021-01-21
#==========================================#
import os, glob
import csv
import numpy as np
import cv2

import random


class DataLoader():

    def __init__(self, dataset_name = 'airplane_vs_ship', shuffle=False):
        # Initialize variables
        self.datasets = os.listdir('./datasets')
        self.datasets_name = dataset_name
        self.labels = []
        self.imgs = np.empty((len(self.datasets),200,200,3))

        self.shuffle = shuffle


    def create(self, ratio1=0.7, ratio2=0.9):
        imgs = sorted(glob.glob(os.path.join('./datasets','*.jpg')))

        for idx, img in enumerate(self.datasets):
            temp_label = self.datasets[idx].split('_')[0]
            self.labels.append(temp_label)

            img = cv2.imread('./datasets/'+img, cv2.IMREAD_COLOR)
            resized_img = cv2.resize(img, (200,200), interpolation=cv2.INTER_AREA)
            self.imgs[idx] = resized_img

        data = self.imgs
        labels = self.labels

        if self.shuffle:
            shuffle = np.arange(len(self.datasets))
            np.random.shuffle(shuffle)

            data = self.imgs[shuffle]
            labels = self.labels[shuffle]
        
        split1 = int(ratio1 * data.shape[0])
        split2 = int(ratio2 * data.shape[0])
        [train_set, validation_set, test_set] = np.split(data, [split1,split2])
        [train_label, validation_label, test_label] = np.split(labels, [split1,split2])
        
        np.savez(os.path.join('./results', 'train'+'.npz'), data = train_set, label=train_label)
        np.savez(os.path.join('./results', 'validation'+'.npz'), data = validation_set, label=validation_label)
        np.savez(os.path.join('./results', 'test'+'.npz'), data = test_set, label=test_label)


    def load(self):
        train_set = np.load('./results/'+'train'+'.npz')
        validation_set = np.load('./results/'+'validation'+'.npz')
        test_set = np.load('./results/'+'test'+'.npz')

        print(train_set['data'][0])
        print(train_set['label'][0])

        return {
            'train_data' : train_set['data'],
            'train_label' : train_set['label']
        }


if __name__ == "__main__":
    dl = DataLoader()
    dl.create()
    dl.load()
