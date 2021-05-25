import numpy as np
import cv2
import os, glob
import cv2
import matplotlib.pyplot as plt


train_set = np.load('./results/'+'train'+'.npz')

plt.imshow(train_set['data'][0]/255)
plt.show()