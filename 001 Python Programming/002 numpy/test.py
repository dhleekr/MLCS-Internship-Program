import numpy as np

test = np.load("train.npz")
x = test['data']
print(x.shape)
