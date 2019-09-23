from anchors import *

from keras import layers, models
from keras import initializers
from keras import regularizers
import keras.backend as K

import numpy as np
import tensorflow as tf 


class RPN:
    def __init__(self, cfg,):
        self.n_channels = cfg.N_CHANNELS
        self.anchors = generate_anchors(cfg.BASE_SIZE, cfg.ANCHOR_RATIO, cfg.ANCHOR_SCALE)
        self.n_anchors = self.anchors.shape[0]
        self.cfg = cfg

    def __call__(self, ):
        inp = layers.Input((None,None, self.n_channels))
        map_size = tf.shape(inp)[1:3]
        shared = layers.Conv2D(self.n_channels, 3, activation='relu', padding='SAME',
                        kernel_initializer=initializers.random_normal(stddev=0.01),
                        kernel_regularizer=regularizers.l2(1.0),
                        bias_regularizer=regularizers.l2(2.0))(inp)

        logits = layers.Conv2D(self.n_anchors*2, 1, 
                            kernel_initializer=initializers.random_normal(stddev=0.01),
                            kernel_regularizer=regularizers.l2(1.0),
                            bias_regularizer=regularizers.l2(2.0))(shared)
        logits = layers.Reshape((-1, 2))(logits)

        score = layers.Softmax()(logits)

        delta = layers.Conv2D(self.n_anchors*4,  1, 
                            kernel_initializer=initializers.random_normal(stddev=0.01),
                            kernel_regularizer=regularizers.l2(1.0),
                            bias_regularizer=regularizers.l2(2.0))(shared)
        delta = layers.Reshape((-1,4))(delta)
        
        
        model = models.Model(inp, [logits, delta, score])
        return model


class Proposal_layer(layers.Layer):
    def __init__(self, img_size, map_size, scale, cfg, mode, **kwargs):
        super(Proposal_layer, self).__init__(**kwargs)
        self.img_size = img_size
        self.map_size = map_size
        self.scale = scale
        self.cfg = cfg
        self.anchors = generate_anchors(cfg.BASE_SIZE, cfg.ANCHOR_RATIO, cfg.ANCHOR_SCALE)
        self.n_anchors = self.anchors.shape[0]
        if mode == 'inference': 
            self.post_nms_n = cfg.POST_NMS_N_INFER
            self.pre_nms_n = cfg.PRE_NMS_N_INFER
        else: 
            self.post_nms_n = cfg.POST_NMS_N
            self.pre_nms_n = cfg.PRE_NMS_N
            
    # delta predict
    def call(self, inputs, **kwargs):
        delta, score = inputs
        score = score[0,:,1]       
        
        all_anchors = shift(self.anchors, self.cfg.FEAT_STRIDE, self.map_size)    
        
        proposals = bbox_transform_inv(all_anchors, delta[0])      
        proposals = ClipBbox()([proposals, self.img_size * self.scale[0]])

        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]

        keep = tf.cast(tf.logical_and(w>=self.cfg.RPN_MIN_SIZE,h>=self.cfg.RPN_MIN_SIZE), tf.int32)
        
        proposals = tf.gather(proposals,keep)
        score = tf.gather(score, keep)

        k = tf.shape(score)[0]
        score, keep = tf.cond(self.pre_nms_n<=k, lambda: tf.nn.top_k(score, self.pre_nms_n), lambda: tf.nn.top_k(score, k))
        proposals = tf.gather(proposals, keep)

        keep = tf.image.non_max_suppression(proposals, score, self.post_nms_n, self.cfg.NMS_THRESH)
        proposals = tf.gather(proposals,keep)
        
        pad = tf.maximum(self.cfg.POST_NMS_N - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, ([0,pad],[0,0]))

        return proposals[None,]

        
    def compute_output_shape(self, input_shape):
        return (None, self.cfg.POST_NMS_N, 4)

class ClipBbox(layers.Layer):
    def call(self, inputs, **kwargs):
        x , img_size = inputs
        
        img_size = tf.cast(img_size, tf.float32)
        xlim, ylim = img_size[0], img_size[1]
        x0 = tf.clip_by_value(x[:, 0], 0, xlim)
        y0 = tf.clip_by_value(x[:, 1], 0, ylim)
        x1 = tf.clip_by_value(x[:, 2], 0, xlim)
        y1 = tf.clip_by_value(x[:, 3], 0, ylim)

        return tf.stack([x0, y0, x1, y1],axis=1)
    
    def compute_output_shape(self, input_shape):
        return input_shape[1]    





class Proposal_Target_layer(layers.Layer):
    def __init__(self, cfg, mean=None, std=None, **kwargs):
        super(Proposal_Target_layer, self).__init__()
        self.cfg = cfg
        self.mean = mean
        self.std = std

    def call(self, inputs, **kwargs):
        rois = inputs[0][0]
        gt_bbox = inputs[1][0]
        labels = inputs[2][0]
        
        _, rois = self._trim_pad(rois)

        all_rois = tf.concat([rois, gt_bbox], axis=0)

        ious = compute_overlaps_tf(all_rois, gt_bbox)

        argmax_row = tf.argmax(ious, axis=1)
        labels = tf.gather(labels, argmax_row)
        
        max_row = tf.reduce_max(ious, axis=1)
        
        idxs_fg = tf.where(max_row > self.cfg.FG_THRESH)
        n_fg = tf.minimum(self.cfg.N_FG_ROIS, tf.shape(idxs_fg)[0])        
        idxs_fg = tf.cond(tf.shape(idxs_fg)[0]>0, lambda:self._choice(idxs_fg,n_fg),lambda: idxs_fg)

        idxs_bg = tf.where(tf.logical_and((max_row>self.cfg.BG_THRESH_LO),(max_row<self.cfg.BG_THRESH_HI)))
        n_bg = self.cfg.N_ROIS - n_fg 
        idxs_fg = tf.cond(tf.shape(idxs_bg)[0]>0, lambda:self._choice(idxs_bg,n_bg),lambda: idxs_bg)
        
        idxs = tf.concat([idxs_fg, idxs_bg], axis=0)[:, 0]
        rois = tf.gather(all_rois, idxs)
        labels = tf.gather(labels, idxs)

        gt_idxs = tf.gather(argmax_row, idxs)
        gt_bbox = tf.cast(tf.gather(gt_bbox, gt_idxs), tf.float32)
        delta2_ = bbox_transform_tf(rois, gt_bbox)
        
        if not self.mean: self.mean = np.array([0., 0., 0., 0.])
        if not self.std: self.std = np.array([0.2, 0.2, 0.2, 0.2])        
        delta2_ *= self.mean
        delta2_ /= self.std

        return [rois, delta2_, labels, idxs_fg]


            
    def _trim_pad(self, rois):
        non_zero = tf.equal(tf.reduce_sum(tf.abs(rois), axis=1), 0)
        rois = tf.boolean_mask(rois, non_zero)
        return non_zero, rois
    
    def _choice_replacement(self, inputs, n_samples):
        log = tf.expand_dims(tf.zeros(tf.shape(inputs)[0]), 0)
        idx = tf.multinomial(log, n_samples)
        idx = tf.squeeze(idx, 0)
        return tf.gather(inputs, idx)    

    def _choice(self, inputs, n_samples):
        n = tf.shape(inputs)[0]
        idx = tf.random_shuffle(tf.range(n))[:n_samples]
        return tf.gather(inputs, idx) 

    def compute_output_shape(self, input_shape):
        n_rois = self.cfg.N_ROIS
        return [(n_rois,4), (n_rois, 4), (n_rois,1), (None,1)]



class RoiPooling(layers.Layer):
    def __init__(self, pool_width=7, pool_height=7,):
        super(RoiPooling, self).__init__()
        self.pool_width = pool_width
        self.pool_height = pool_height

    def call(self, inputs):
        """[summary]
        
        Arguments:
            x {[type]} -- [1, H, W, C]
            rois {[type]} -- [N_ROIS, 4]
        Returns:
            [type] -- [1, N_ROIS, pool_heigt, pool_widtd, C]
        """
        x, rois = inputs


        def fn(inp):
            # H, W
            x_shape = tf.shape(x)[1:3][::-1]
            scale = tf.cast(tf.tile(x_shape, (2,)), tf.float32)
            roi = tf.cast(inp / scale, tf.int32)
            w_stride = (roi[2] - roi[0]) // self.pool_width
            h_stride = (roi[3] - roi[1]) // self.pool_height
#             roi_pool = tf.image.resize_images(x[0, roi[1]:roi[3], roi[0]:roi[2],:], [self.pool_height,self.pool_width],)            
            roi_pool = []
            for i in range(self.pool_height):
                h_start = roi[1] + i * h_stride
                for j in range(self.pool_width):
                    w_start = roi[0] + j * w_stride
                    val = tf.reduce_max(x[0, h_start:h_start+h_stride, w_start:w_start+w_stride, :], axis=[0,1])
                    roi_pool.append(val)
            roi_pool = tf.reshape(tf.stack(roi_pool,axis=0), (self.pool_height,self.pool_width, -1))
            return roi_pool
        rois_pool = tf.map_fn(fn, rois)
        return [rois_pool[None,]]#, non_zero]

    def compute_output_shape(self, input_shape):
        x_shape, rois_shape = input_shape
        return [(rois_shape[0], rois_shape[1], self.pool_height, self.pool_width, x_shape[-1]), ]#(None,1)]
    





class RoiAlian(layers.Layer):
    def __init__(self, pool_height, pool_width):
        self.pool_height = pool_height
        self.pool_width = pool_width

    def call(self, inp, **args):
        x, rois = inp
        x_shape = x.shape.as_list()[1:3]
        # x = tf.pad(x, [0,0],[1,1],[1,1],[0,0], method='SYMMETRIC')
        rois += 1

        x0,y0,x1,y1 = tf.split(rois, 4, 1)
        w = x1 - x0
        h = y1 - y0

        xx = (x0 + w/2/self.pool_width - .5) / x_shape[1]
        yy = (y0 + h/2/self.pool_height - .5) / x_shape[0]

        w /= x_shape[1]
        h /= x_shape[0]
        rois =  tf.concat([xx, yy, xx+w, yy+h], axis=1)
        roi_pools = tf.image.crop_and_resize(x, rois, rois, tf.zeros([tf.shape(rois)[0]], dtype=tf.int32), 
                                             [self.pool_width, self.pool_height])
        roi_pools = tf.nn.avg_pool(rois, [1,1,2,2,], [1,1,2,2], padding="SAME")
        return roi_pools

    def compute_output_shape(self, input_shape):
        x, rois = input_shape
        return (None,) + x[1:]


