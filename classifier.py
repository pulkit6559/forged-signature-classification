import keras
import sys
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from keras.models import Model, Sequential
from keras.engine import InputSpec, Layer
from keras import regularizers
from keras.optimizers import SGD, Adam
from keras.utils.conv_utils import conv_output_length
from keras import activations
import numpy as np

a = Input((200, 400, 1))
b = Input((200, 400, 1))

model = Sequential()
model.add(Conv2D(kernel_size=(5, 5), filters=20,input_shape=(200, 400, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(kernel_size=(5, 5), filters=25, activation='relu'))
model.add(MaxPooling2D((2, 2)))


feat_map1 = model(b)
feat_map2 = model(a)
