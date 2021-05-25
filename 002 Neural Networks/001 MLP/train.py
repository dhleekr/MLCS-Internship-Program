import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.models import load_model
import numpy as np

batch_size = 64
num_classes = 10
epochs1 = 20
epochs2 = 20
ratio1 = 0.7
ratio2 = 0.85


# the data, split between train and test sets
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
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# # convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

""" 
model 1
"""
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs1,
                    verbose=1,
                    validation_data=(x_test, y_test))

model.save('mnist1.h5')

"""  
model 2
"""
model2 = Sequential()
model2.add(Dense(1024, activation='relu', input_shape=(784,)))
model2.add(Dropout(0.5))
model2.add(Dense(512,activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(num_classes, activation='softmax'))

model2.summary()

model2.compile(loss='categorical_crossentropy', 
               optimizer=RMSprop(),
               metrics=['accuracy'])

history2 = model2.fit(x_train,y_train,
                      batch_size=batch_size,
                      epochs=epochs2,
                      verbose=1,
                      validation_data=(x_test, y_test))

model2.save('mnist2.h5')