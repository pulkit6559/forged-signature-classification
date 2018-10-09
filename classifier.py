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
