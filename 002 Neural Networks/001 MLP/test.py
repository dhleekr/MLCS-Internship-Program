import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
import numpy as np

num_classes = 10
ratio1 = 0.7
ratio2 = 0.85

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.concatenate([x_train, x_test], axis=0)
y = np.concatenate([y_train, y_test], axis=0)

[x_train, x_valid, x_test] = np.split(x, [int(len(x)*ratio1), int(len(x)*ratio2)])
[y_train, y_valid, y_test] = np.split(y, [int(len(y)*ratio1), int(len(y)*ratio2)])

x_train = x_train.reshape(-1, 784)
x_valid = x_valid.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model1 = load_model('mnist1.h5')
model2 = load_model('mnist2.h5')

score = model1.evaluate(x_test, y_test, verbose=0)
score2 = model2.evaluate(x_test, y_test, verbose=0)

print('Model1 Test loss:', score[0])
print('Model1 Test accuracy:', score[1])

print('Model2 Test loss:', score2[0])
print('Model2 Test accuracy:', score2[1])