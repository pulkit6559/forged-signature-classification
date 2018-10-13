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

normalized_layer = Normalized_Correlation_Layer(stride=(1, 1), patch_size=(5, 5))([feat_map1, feat_map2])

final_layer = Conv2D(kernel_size=(1, 1), filters=25,activation='relu')(normalized_layer)
final_layer = Conv2D(kernel_size=(3, 3), filters=25,activation=None)(final_layer)
final_layer = MaxPooling2D((2, 2))(final_layer)
final_layer = Dense(500)(final_layer)
final_layer = Dense(2, activation="softmax")(final_layer)

#creating a custom normalized layer layer
class Normalized_Correlation_Layer(Layer):
#create a class inherited from keras.engine.Layer.

	def __init__(self, patch_size=(5, 5),
          dim_ordering='tf',
          border_mode='same',
          stride=(1, 1),
          activation=None,
          **kwargs):

       if border_mode != 'same': 
          raise ValueError('Invalid border mode for Correlation Layer ' 
                     '(only "same" is supported as of now):', border_mode) 
       self.kernel_size = patch_size 
       self.subsample = stride 
       self.dim_ordering = dim_ordering 
       self.border_mode = border_mode 
       self.activation = activations.get(activation) 
       super(Normalized_Correlation_Layer, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return(input_shape[0][0], input_shape[0][1], input_shape[0][2], self.kernel_size[0] * input_shape[0][2]*input_shape[0][-1])

    def get_config(self): 
        config = {'patch_size': self.kernel_size, 
          'activation': self.activation.__name__, 
          'border_mode': self.border_mode, 
          'stride': self.subsample, 
          'dim_ordering': self.dim_ordering} 
        base_config = super(Correlation_Layer, self).get_config() 
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, x, mask=None):
        input_1, input_2 = x 
        stride_row, stride_col = self.subsample 
        inp_shape = input_1._keras_shape
        padding_row = (int(self.kernel_size[0] / 2),int(self.kernel_size[0] / 2)) 
        padding_col = (int(self.kernel_size[1] / 2),int(self.kernel_size[1] / 2)) 
        input_1 = K.spatial_2d_padding(input_1, padding =(padding_row,padding_col)) 
        input_2 = K.spatial_2d_padding(input_2, padding = ((padding_row[0]*2, padding_row[1]*2),padding_col)) 




