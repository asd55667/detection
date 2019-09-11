from anchors import *

from keras import layers, models
from keras import initializers
from keras import regularizers

import numpy as np
import tensorflow as tf 

from cfg import cfg

class RPN:
    def __init__(self, img_size, cfg):
        self.img_size = img_size
        self.anchors = generate_anchors(cfg.BASE_SIZE, cfg.ANCHOR_RATIO, cfg.ANCHOR_SCALE)
        self.n_anchors = self.anchors.shape[0]
        self.batch_size = cfg.RPN_BATCH_SIZE


    def __call__(self, x):
        # x (N, H, W, C)
        x_shape = x.shape.as_list()
        self.map_size = x_shape[1:3]
        # inp = layers.Input((None,None,x_shape[-1]))
        shared = layers.Conv2D(x_shape[-1], 3, activation='relu', padding='SAME',
                        kernel_initializer=initializers.random_normal(stddev=0.01),
                        kernel_regularizer=regularizers.l2(1.0),
                        bias_regularizer=regularizers.l2(2.0))(x)

        logits = layers.Conv2D(self.n_anchors*2, 1, 
                            kernel_initializer=initializers.random_normal(stddev=0.01),
                            kernel_regularizer=regularizers.l2(1.0),
                            bias_regularizer=regularizers.l2(2.0))(shared)
        logits = layers.Lambda(lambda x: tf.reshape(x, (-1,2)))(logits)

        score = layers.Softmax()(logits)

        delta = layers.Conv2D(self.n_anchors*4,  1, 
                            kernel_initializer=initializers.random_normal(stddev=0.01),
                            kernel_regularizer=regularizers.l2(1.0),
                            bias_regularizer=regularizers.l2(2.0))(shared)
        delta = layers.Lambda(lambda x: tf.reshape(x, (-1,4)))(delta)
        proposals, scores = Proposal_layer(self.img_size, self.map_size)([delta, score])

        return logits, proposals, scores

    def anchor_target(self, anchors, gt_bbox, img_size):
        # gt_delta
        all_anchors = shift(self.anchors, cfg.FEAT_STRIDE, self.map_size)

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

        labels[max_anchors < cfg.NEG_THRESH] = 0
        labels[argmax_gt] = 1
        labels[max_anchors > cfg.POS_THRESH] = 1
        
        pos_idxs = np.where(labels == 1)[0]
        neg_idxs = np.where(labels == 0)[0]
        
        n_fg = int(self.batch_size * cfg.FG_RATIO)
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

    def proposal_target(self, rois, gt_bbox, labels):
        
        all_bbox = np.vstack(rois, gt_bbox)

        ious = compute_overlaps(np.ascontiguousarray(all_bbox, dtype=np.float32),
                                np.ascontiguousarray(gt_bbox, dtype=np.float32))
    
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


    def _unmap(self, data, rows, idxs, fill):
        if len(data.shape) == 1:
            all_data = np.empty((rows,), dtype=np.float32)
        else:
            all_data = np.empty((rows,) + data.shape[1:], dtype=np.float32)
        all_data.fill(fill)
        all_data[idxs,:] = data
        return all_data   



class Proposal_layer(layers.Layer):
    def __init__(self, img_size, map_size, cfg, *arg, **kwargs):
        self.img_size = img_size
        self.map_size = map_size
        self.cfg = cfg
        self.anchors = generate_anchors(cfg.BASE_SIZE, cfg.ANCHOR_RATIO, cfg.ANCHOR_SCALE)
        self.n_anchors = self.anchors.shape[0]

        super(Proposal_layer, self).__init__(*arg, **kwargs)

    # delta predict
    def call(self, inputs, **kwargs):
        delta, score = inputs
        delta = tf.reshape(delta, (-1,4))
        score = score[:, :, :, self.n_anchors:]       
        score = tf.reshape(score,(-1,))

        all_anchors = shift(self.anchors, self.cfg.FEAT_STRIDE, self.map_size)    
        
        proposals = bbox_transform_inv(all_anchors, delta)      
        proposals = clip_bbox(proposals, self.img_size)

        keep = filter_bboxs(proposals, self.cfg.RPN_MIN_SIZE)
        proposals = tf.gather(proposals,keep)
        score = tf.gather(score, keep)

        keep = tf.nn.top_k(score, self.cfg.PRE_NMS_N)
        proposals = tf.gather(proposals, keep.indices)
        score = keep.values

        keep = tf.image.non_max_suppression(proposals, score, self.cfg.POST_NMS_N, self.cfg.NMS_THRESH)
        proposals = tf.gather(proposals,keep)
        score = tf.gather(score,keep)
        
        pad = tf.maximum(self.cfg.POST_NMS_N - tf.shape(proposals)[0], 0)
        proposals = tf.pad(proposals, [(0,pad),(0,0)])
        score = tf.pad(proposals, [(0,pad),(0,0)])

        return [proposals, score]

        
    def compute_output_shape(self. input_shape):
        return ((None, self.cfg.POST_NMS_N, 4), (None, self.cfg.POST_NMS_N) )
