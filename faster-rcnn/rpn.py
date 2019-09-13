from anchors import *

from keras import layers, models
from keras import initializers
from keras import regularizers
import keras.backend as K

import numpy as np
import tensorflow as tf 

from cfg import cfg

class RPN:
    def __init__(self, cfg):
        self.n_channels = cfg.N_CHANNELS
        self.anchors = generate_anchors(cfg.BASE_SIZE, cfg.ANCHOR_RATIO, cfg.ANCHOR_SCALE)
        self.n_anchors = self.anchors.shape[0]

    def __call__(self, ):
        inp = layers.Input((None,None, self.n_channels))

        shared = layers.Conv2D(self.n_channels, 3, activation='relu', padding='SAME',
                        kernel_initializer=initializers.random_normal(stddev=0.01),
                        kernel_regularizer=regularizers.l2(1.0),
                        bias_regularizer=regularizers.l2(2.0))(inp)

        logits = layers.Conv2D(self.n_anchors*2, 1, 
                            kernel_initializer=initializers.random_normal(stddev=0.01),
                            kernel_regularizer=regularizers.l2(1.0),
                            bias_regularizer=regularizers.l2(2.0))(shared)
        logits = layers.Lambda(lambda x: tf.reshape(x, (K.shape(x)[0],-1,2)))(logits)

        score = layers.Softmax()(logits)

        delta = layers.Conv2D(self.n_anchors*4,  1, 
                            kernel_initializer=initializers.random_normal(stddev=0.01),
                            kernel_regularizer=regularizers.l2(1.0),
                            bias_regularizer=regularizers.l2(2.0))(shared)
        delta = layers.Lambda(lambda x: tf.reshape(x, (K.shape(x)[0],-1,4)))(delta)
        model = models.Model(inp, [logits, delta, score])
        return model


class Proposal_layer(layers.Layer):
    def __init__(self, img_size, map_size, cfg, **kwargs):
        super(Proposal_layer, self).__init__(**kwargs)
        self.img_size = img_size
        self.map_size = map_size
        self.cfg = cfg
        self.anchors = generate_anchors(cfg.BASE_SIZE, cfg.ANCHOR_RATIO, cfg.ANCHOR_SCALE)
        self.n_anchors = self.anchors.shape[0]
        
    # delta predict
    def call(self, inputs, **kwargs):
        delta, score = inputs
        score = score[0,:,1]       
        
        all_anchors = shift(self.anchors, self.cfg.FEAT_STRIDE, self.map_size)    
        
        proposals = bbox_transform_inv(all_anchors, delta[0])      
        proposals = ClipBbox()([proposals, self.img_size])

        w = proposals[:, 2] - proposals[:, 0]
        h = proposals[:, 3] - proposals[:, 1]

        keep = tf.reshape(tf.where(tf.logical_and(w>=self.cfg.RPN_MIN_SIZE,h>=self.cfg.RPN_MIN_SIZE)),[-1])
        
        proposals = tf.gather(proposals,keep)
        score = tf.gather(score, keep)

        score, keep = tf.nn.top_k(score, self.cfg.PRE_NMS_N)
        proposals = tf.gather(proposals, keep)

        keep = tf.image.non_max_suppression(proposals, score, self.cfg.POST_NMS_N, self.cfg.NMS_THRESH)
        proposals = tf.gather(proposals,keep)
        
        pad = tf.maximum(self.cfg.POST_NMS_N - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, ([0,pad],[0,0]))

        return proposals[None,]

        
    def compute_output_shape(self, input_shape):
        return (None, self.cfg.POST_NMS_N, 4)



class RoiPooling(layers.Layer):
    def __init__(self, pool_width=7, pool_height=7,):
        super(RoiPooling, self).__init__()
        self.pool_width = pool_width
        self.pool_height = pool_height

    def call(self, inp):
        """[summary]
        
        Arguments:
            x {[type]} -- [1, H, W, C]
            rois {[type]} -- [POST_NMS_N, 4]
        Returns:
            [type] -- [1, POST_NMS_N, pool_heigt, pool_widtd, C]
        """
        x, rois = inp
        # _, rois = self._trim_pad(rois)

        def fn(inp):
            # H, W
            x_shape = K.shape(x)[1:3][::-1]
            scale = tf.cast(tf.tile(x_shape, (2,)), tf.float32)
            roi = tf.cast(inp[0]/scale, tf.int32)
            w_stride = (roi[2] - roi[0]) // self.pool_width
            h_stride = (roi[3] - roi[1]) // self.pool_height
#             roi_pool = tf.image.resize_images(x[0, roi[1]:roi[3], roi[0]:roi[2],:], [self.pool_height,self.pool_width],)            
            roi_pool = []
            for i in range(self.pool_height):
                start = roi[0] + i * h_stride
                for j in range(self.pool_width):
                    val = tf.reduce_max(x[0,start:start+i*h_stride, start+(j+1)*w_stride,:],axis=0)
                    roi_pool.append(val)
            roi_pool = tf.reshape(tf.stack(roi_pool,axis=0), (self.pool_height,self.pool_width,-1))
            return roi_pool
        rois_pool = tf.map_fn(fn, rois[0])

        return rois_pool[None,]
    
    def _trim_pad(self, rois):
        non_zero = tf.equal(tf.reduce_sum(tf.abs(rois), axis=1), 0)
        rois = tf.boolean_mask(rois, non_zero)
        return non_zero, rois

    def compute_output_shape(self, input_shape):
        x_shape, rois_shape = input_shape
        return rois_shape[:2] + (self.pool_height, self.pool_width, x_shape[-1])  
    



class RoiAlian(layers.Layer):
    def __init__(self, pool_height, pool_width):
        self.pool_height = pool_height
        self.pool_width = pool_width

    def call(self, inp, **args):
        x, rois = inp
        x_shape = x.shape.as_list()[1:3]
        x = tf.pad(x, [0,0],[1,1],[1,1],[0,0], method='SYMMETRIC')
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
        roi_pools = tf.nn.avg_pool(ret, [1,1,2,2,], [1,1,2,2], padding="SAME")
        return roi_pools

    def compute_output_shape(self, input_shape):
        x, rois = inp
        return (None,) + x[1:]



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
# TODO



class Anchor_Target_layer(layers.Layer):
    # (self, anchors, gt_bbox, img_size)
        # gt_delta
    def __init__(self, ):
        all_anchors = shift(self.anchors, self.cfg.FEAT_STRIDE, self.map_size)

        idxs_inside = np.where(all_anchors[:,0] >= 0 & 
                            all_anchors[:,1] >= 0 &
                            all_anchors[:,2] <= W &
                            all_anchors[:,3] <= H)[0]

        anchors = all_anchors[idxs_inside, :]    
        
        ious = compute_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
                                np.ascontiguousarray(gt_bbox, dtype=np.float32))

        argmax_anchors = ious.argmax(axis=1)
        argmax_gt = ious.argmax(axis=0)
        max_anchors = ious[np.arange(ious.shape[0]), argmax_anchors]
        max_gt = ious[argmax_gt, np.arange(ious.shape[1])]

        labels = -1 * np.ones((ious.shape[0]), dtype=np.float32)

        labels[max_anchors < self.cfg.NEG_THRESH] = 0
        labels[argmax_gt] = 1
        labels[max_anchors > self.cfg.POS_THRESH] = 1
        
        pos_idxs = np.where(labels == 1)[0]
        neg_idxs = np.where(labels == 0)[0]
        
        n_fg = int(self.batch_size * self.cfg.FG_RATIO)
        if len(pos_idxs) > n_fg:
            disabled = np.random.choice(pos_idxs, size=len(pos_idxs) - n_fg, replace=False)
            labels[disabled] = -1

        n_bg = self.batch_size - n_fg
        if len(neg_idxs) > n_bg:
            disabled = np.randon.choice(neg_idxs, size=len(neg_idxs) - n_bg, replace=False)
            label[disabled] = -1

        delta_ = bbox_transform(anchors, gt_bbox[argmax_anchors,:])
        
        labels = self._unmap(labels, all_anchors.shape[0], idxs_inside, -1)
        delta_ = self._unmap(delta_, all_anchors.shape[0], idxs_inside, 0)

        return delta_, labels



class Proposal_Target_layer(layers.Layer):
#    (self, rois, gt_bbox, labels)
    def __init__(self,):
        all_bbox = tf.stack([rois, gt_bbox], axis=0)
        gt_bbox = tf.identity(gt_bbox)
        ious = compute_overlaps(all_bbox, gt_bbox)

        argmax_row = ious.argmax(1)
        max_row = np.max(ious,1)

        idxs_fg = np.where(max_row > self.fg_thresh)[0]
        n_fg = rois.shape[1] * FG_RATIO
        if len(idxs_fg) > n_fg:
            idxs_fg = np.random.choice(idxs_fg, size=n_fg, replace=False)
        
        idxs_bg = np.where((max_row>self.bg_thresh_low) & (max_row<self.bg_thresh_high))[0]
        n_bg = rois.shape[1] - n_fg
        if len(idxs_bg) > n_bg:
            idxs_bg = np.random.choice(idxs_bg, size=n_bg, replace=False)
        
        idxs = np.append(idxs_fg, idxs_bg)
        labels = labels[idxs]
        bbox = all_bbox[idxs]

        delta2_ = bbox_transform(bbox, gt_bbox)
        delta2_ *= np.array(self.mean)
        delta2_ /= np.array(self.std)
        return delta2_, labels

        def _unmap(self, data, rows, idxs, fill):
            if len(data.shape) == 1:
                all_data = np.empty((rows,), dtype=np.float32)
            else:
                all_data = np.empty((rows,) + data.shape[1:], dtype=np.float32)
            all_data.fill(fill)
            all_data[idxs,:] = data
            return all_data   
