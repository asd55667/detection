# from cfg import *
# from rpn import RPN

from keras import layers, models
from keras import initializers
from keras import regularizers
import keras.backend as K

import tensorflow as tf



# class RoiPooling(layers.Layer):
#     def __init__(self, pool_width=7, pool_height=7, feat_stride=16):
#         super(RoiPooling, self).__init__()
#         self.pool_width = pool_width
#         self.pool_height = pool_height
#         self.scale = 1. / feat_stride

#     def call(self, inp):
#         # x (1, 16, 16, 512)
#         # rois (N, 4)
#         feature_map, rois = inp

#         rois *= self.scale

#         # w_stride = (rois[:,2::4] - rois[:,0::4]) / self.pool_width
#         # h_stride = (rois[:,3::4] - rois[:,1::4] ) / self.pool_height
#         # w_stride = K.cast(K.maximum(w_stride,1), 'int32')
#         # h_stride = K.cast(K.maximum(h_stride,1), 'int32')

#         # rois_pool = K.zeros((1, rois.shape[0], self.pool_width, self.pool_height, feature_map.shape[-1]))
#         # for i, roi in enumerate(rois):
#         #     x1 = K.cast(roi[0], 'int32')
#         #     y1 = K.cast(roi[1], 'int32')

#         #     for row in range(self.pool_width):
#         #         for col in range(self.pool_height):
#         #             patch = feature_map[:,x1:x1+w_stride[row], y1:y1+h_stride[col], :]
#         #             max_path = K.max(patch, (1,2))[None,:,:]
#         #             rois_pool[0,i, row, col, :] = max_path

#         #
#         rois_pool = []
#         for roi in rois:

#             x1 = K.cast(roi[0], 'int32')
#             y1 = K.cast(roi[1], 'int32')
#             x2 = K.cast(roi[2], 'int32')
#             y2 = K.cast(roi[3], 'int32')

#             roi_pool = tf.image.resize_images(feature_map[0, x1:x2, y1:y2,:], (self.pool_width,self.pool_height))
#             rois_pool.append(roi_pool)

#         rois_pool = K.concatenate(rois_pool, 0)
#         rois_pool = K.expand_dims(rois_pool, 0)
        
#         return rois_pool     

#     def compute_output_shape(self, inp):
#         feature_map, rois = inp
#         return None, rois.shape[0], self.pool_width, self.pool_height, feature_map.shape[-1] 


class RoiPooling(layers.Layer):
    def __init__(self, pool_width=7, pool_height=7,):
        super(RoiPooling, self).__init__()
        self.pool_width = pool_width
        self.pool_height = pool_height

    def call(self, inp):
        # x (1, H, W, C)
        # rois (N, 4)
        x, rois = inp

        x_shape = x.shape.as_list()[1:3][::-1]
        scale = tf.Variable(x_shape+x_shape, dtype=tf.float32)        
        rois = tf.cast(rois / scale, tf.int32)
        tf.image.crop_and_resize(x,rois, tf.zeros([tf.shape(rois)[0]], dtype=tf.int32), [self.pool_height, self.pool_width])

        # rois_pool = tf.map_fn(self.helper, rois)

        return rois_pool     
    
    def compute_output_shape(self, inp):
        x, rois = inp
        return (1,None)+ x[1:]
    
    def helper(self, inp):
        x_shape = x.shape.as_list()[1:3][::-1]
        scale = tf.Variable(x_shape+x_shape, dtype=tf.float32)
        roi = tf.cast(inp[0] / scale, tf.int32)
        roi_pool = tf.image.resize_images(x[0,roi[0]:roi[2], roi[1]:roi[3],:], [self.pool_height,self.pool_width],)
        return roi_pool        


class Detector():

    def __init__(self, n_classes, img_size,
                backbone, head):
        self.n_classes = self.n_classes
        self.backbone = backbone
        self.rpn = RPN(img_size)
        self.stage2 = Head(self.n_classes)

    def __call__(self, inp, ):

        x = self.backbone(inp)
        rois = self.rpn(x)
        x = RoiPooling()([x, rois])
        bboxes, scores = self.stage2(x)
        return bboxes, scores
        
    

    










import numpy as np
import tensorflow as tf


def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.

    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
        boxes = boxes + 1

    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
        spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

        nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
        nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(image)[2:]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])   # nhwc
    ret = tf.image.crop_and_resize(
        image, boxes, tf.to_int32(box_ind),
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
    return ret


def roi_align(featuremap, boxes, resolution):
    # sample 4 locations per roi bin
    ret = crop_and_resize(
        featuremap, boxes,
        tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        resolution * 2)
    ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return ret


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

    def compute_output_shape(self, inp):
        x, rois = inp
        return (None,) + x[1:]











# if __name__ == '__main__':
#     import tensorflow.contrib.eager as tfe
#     tfe.enable_eager_execution()

#     # want to crop 2x2 out of a 5x5 image, and resize to 4x4
#     image = np.arange(25).astype('float32').reshape(5, 5)
#     boxes = np.asarray([[1, 1, 3, 3]], dtype='float32')
#     target = 4

#     print(crop_and_resize(
#         image[None, None, :, :], boxes, [0], target)[0][0])
#     """
#     Expected values:
#     4.5 5 5.5 6
#     7 7.5 8 8.5
#     9.5 10 10.5 11
#     12 12.5 13 13.5
#     """        