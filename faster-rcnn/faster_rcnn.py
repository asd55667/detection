# from anchors import *
from cfg import cfg
from rpn import *

from keras import layers, models
from keras import initializers
from keras import regularizers
import keras.backend as K


import tensorflow as tf
import functools

class Detector:
    def __init__(self, backbone, head, cfg):
        self.n_classes = cfg.N_CLASSES
        self.backbone = backbone
        self.rpn = RPN(cfg)()
        self.head = functools.partial(head, cfg=cfg)
        self.model = self.build_model()

    def build_model(self,):
        inp = layers.Input((None,None,3))
        img_size = K.shape(inp)[1:3]

        x = self.backbone(inp)
        map_size = K.shape(x)[1:3]

        logits, delta, scores = self.rpn(x)
        rois = Proposal_layer(img_size, map_size, cfg)([delta, scores])

        x_roi = RoiPooling()([x,rois])
        logits2, bbox, scores2 = self.head(x_roi)

        model = models.Model([inp], [score, bbox])

    def predict(self, x):
        self.model.predict(x)        
        
      

def head(x, cfg):
    # x (batch_size=1, rois, pool_height,pool_width, channels)            
    x = layers.TimeDistributed(layers.Conv2D(cfg.FC_LAYERS, (7,7),))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    x = layers.TimeDistributed(layers.Conv2D(cfg.FC_LAYERS, (1,1),))(x)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.ReLU())(x)
    shared = layers.Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2),)(x)
    logits2 = layers.TimeDistributed(layers.Dense(cfg.N_CLASSES))(shared)
    scores2 = layers.TimeDistributed(layers.Softmax())(logits2)
    bbox = layers.TimeDistributed(layers.Dense(4*cfg.N_CLASSES))(shared)
    bbox = layers.Reshape((-1,cfg.N_CLASSES,4))(bbox)
    return logits2, bbox, scores2





