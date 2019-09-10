from anchors_tensor import *

from keras import layers, models
from keras import initializers
from keras import regularizers

import numpy as np

from cfg import *

class RPN:
    def __init__(self, img_size, rpn_batch_size=RPN_BATCH_SIZE):
        self.img_size = img_size
        self.anchors = generate_anchors(16,ANCHOR_SCALE, ANCHOR_RATIO)
        self.n_anchors = self.anchors.shape.as_list()[0]
        self.batch_size = rpn_batch_size


    def __call__(self, x):
        # x (N, H, W, C)
        x_shape = x.shape.as_list()
        self.map_size = x_shape[1:3]
        inp = layers.Input((None,None,x_shape[-1]))
        x_ = layers.Conv2D(x_shape[-1], 3, activation='relu', padding='SAME',
                        kernel_initializer=initializers.random_normal(stddev=0.01),
                        kernel_regularizer=regularizers.l2(1.0),
                        bias_regularizer=regularizers.l2(2.0))(inp)

        score = layers.Conv2D(self.n_anchors*2, 1, activation='softmax',
                            kernel_initializer=initializers.random_normal(stddev=0.01),
                            kernel_regularizer=regularizers.l2(1.0),
                            bias_regularizer=regularizers.l2(2.0))(x_)

        delta = layers.Conv2D(self.n_anchors*4,  1, 
                            kernel_initializer=initializers.random_normal(stddev=0.01),
                            kernel_regularizer=regularizers.l2(1.0),
                            bias_regularizer=regularizers.l2(2.0))(x_)

        proposals, scores = Proposal_layer(self.img_size, self.map_size)([delta, score])

        rpn_model = models.Model(inp, [proposals, scores])
        
        return rpn_model

    def anchor_target(self, anchors, gt_bbox, img_size):
        # gt_delta
        all_anchors = shift(self.anchors, FEAT_STRIDE, self.map_size)

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

        labels[max_anchors < NEG_THRESH] = 0
        labels[argmax_gt] = 1
        labels[max_anchors > POS_THRESH] = 1
        
        pos_idxs = np.where(labels == 1)[0]
        neg_idxs = np.where(labels == 0)[0]
        
        n_fg = int(self.batch_size * FG_RATIO)
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
    def __init__(self, img_size, map_size):
        super(Proposal_layer,self).__init__()
        self.img_size = img_size
        self.map_size = map_size
        self.anchors = generate_anchors(16,ANCHOR_SCALE, ANCHOR_RATIO)
        self.n_anchors = self.anchors.shape.as_list()[0]
        
    # delta predict
    def call(self, inputs, **kwargs):
        delta, score = inputs
        delta = tf.reshape(delta, (-1,4))
        score = score[:, :, :, self.n_anchors:]       
        score = tf.reshape(score,(-1,))

        all_anchors = shift(self.anchors, FEAT_STRIDE, self.map_size)    
        
        proposals = bbox_transform_inv(all_anchors, delta)      
        proposals = clip_bbox(proposals, self.img_size)

        keep = filter_bboxs(proposals, RPN_MIN_SIZE)
        proposals = tf.gather(proposals,keep)
        score = tf.gather(score, keep)

        keep = tf.nn.top_k(score,PRE_NMS_N)
        proposals = tf.gather(proposals, keep.indices)#tf.reshape(keep.indices,[-1]))
        score = keep.values
        #score = tf.gather(score, tf.reshape(keep.indices,[-1]))

        #
        keep = tf.image.non_max_suppression(proposals, score, POST_NMS_N, NMS_THRESH, )
        proposals = tf.gather(proposals,keep)
        score = tf.gather(score,keep)

        return [proposals, score]

        
        