import numpy as np
import tensorflow as tf
import keras.backend as K

def ddx(y):
    #if tf.is_tensor(y):
    res = K_ddx2D(y)
    #else:
    #    res = np_ddx2D(y)
    return res

def ddy(y):
    #if tf.is_tensor(y):
    res = K_ddy2D(y)
    #else:
    #    res = np_ddy2D(y)
    return res

# Assume y shape is (B, H, W, C)
def np_ddx2D(y):
    return np.gradient(y, axis=1) 

# Assume y shape is (B, H, W, C)
def K_ddx2D(y):
    return K.concatenate((
        K.expand_dims(y[:, 1, :, :] - y[:, 0, :, :], axis=1),
        y[:, 1:, :, :] - y[:, :-1, :, :]), axis=1)

# Assume y shape is (B, H, W, C)
def np_ddy2D(y):
    return np.gradient(y, axis=2) 

# Assume y shape is (B, H, W, C)
def K_ddy2D(y):
    return K.concatenate((
        K.expand_dims(y[:, :, 1, :] - y[:, :, 0, :], axis=2),
        y[:, :, 1:, :] - y[:, :, :-1, :]), axis=2)