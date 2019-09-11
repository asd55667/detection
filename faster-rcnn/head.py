# stage 2
import numpy as np 
from anchors_tensor import *
from keras import layers, models
import tensorflow as tf 


class Head:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        

    def __call__(self, x):
        # x (batch_size=1, rois, pool_width, pool_height, channels)
        x_shape = x.shape.as_list()
        batch_size = x_shape[1:]
        rois_extractor = self._build_rois_extractor(x)

        inp = layers.Input((inp_shape))

        x_ = layers.Flatten()(inp)
        x_ = layers.Dense(4096, activation='relu',
                    kernel_regularizer=regularizers.l2(1.0),
                    bias_regularizer=regularizers.l2(2.0))(x_)
        x_ = layers.Dense(4096, activation='relu',
                    kernel_regularizer=regularizers.l2(1.0),
                    bias_regularizer=regularizers.l2(2.0))(x_)
        
        score = layers.Dense(self.n_classes, activation='softmax',
                    kernel_initializer=initializers.random_normal(stddev=0.01))(x_)    
        bbox = layers.Dense(self.n_classes*4, activation='linear',
                    kernel_initializer=initializers.random_normal(stddev=0.01))(x_)
        return score, delta


        
    def _build_rois_extractor(self, x):
        # x (batch_size=1, rois, pool_width, pool_height, channels)
        x_shape = x.shape.as_list()
        inp_shape = x_shape[1:]

        inp = layers.Input((inp_shape))

        x = layers.Flatten()(inp)
        x = layers.Dense(4096, activation='relu',
                    kernel_regularizer=regularizers.l2(1.0),
                    bias_regularizer=regularizers.l2(2.0))(x)
        x = layers.Dense(4096, activation='relu',
                    kernel_regularizer=regularizers.l2(1.0),
                    bias_regularizer=regularizers.l2(2.0))(x)
        rois_extractor = models.Model([inp,x])
        return rois_extractor
    


