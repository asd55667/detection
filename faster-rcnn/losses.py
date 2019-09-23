import tensorflow as tf 
from keras import layers


def smooth_l1_loss(delta_, delta, sigma=3.):        
    x = tf.abs(delta - delta_)
    cond = tf.less(x, 1./sigma**2)
    loss = tf.where(cond, 0.5*sigma**2 * tf.math.pow(x,2), x - .5 / sigma**2)
    return tf.reduce_sum(loss)


class Rpn_class_loss(layers.Layer):
    def __init__(self, name='rpn_class_loss', **kwargs):
        super(Rpn_class_loss, self).__init__(name='rpn_class_loss', **kwargs)

    def call(self, inputs, **kwargs):
        y_, y = inputs
        y_ = y_[0, :,4]
        idxs = tf.where(tf.not_equal(y_, -1))       

        y_ = tf.gather_nd(y_, idxs)

        y = y[0, :, 1]
        y = tf.gather_nd(y, idxs)    
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y)


class Rpn_bbox_loss(layers.Layer):
    def __init__(self, name='rpn_bbox_loss', **kwargs):
        super(Rpn_bbox_loss, self).__init__(name='rpn_bbox_loss', **kwargs)

    def call(self, inputs, **kwargs):
        delta_label, delta = inputs
        delta_ = delta_label[0, :,:4]
        labels = delta_label[0, :, 4]
        
        idxs = tf.where(tf.equal(labels, 1))
        delta_ = tf.gather_nd(delta_, idxs)
        delta = tf.gather_nd(delta[0], idxs)
        
        return smooth_l1_loss(delta_, delta)

class Roi_class_loss(layers.Layer):
    def __init__(self, name='roi_class_loss', **kwargs):
        super(Roi_class_loss, self).__init__(name='roi_class_loss', **kwargs)

    def call(self, inputs, **kwargs):
        """[summary]
        
        Arguments:
            y_ {[type]} -- [n_rois, 1]
            y {[type]} -- [batch_size=1, n_rois, n_classes]    
        """        
        y_, y = inputs
        y = y[0]
        y_ = tf.cast(y_[:,0], tf.int32)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)

class Roi_bbox_loss(layers.Layer):
    def __init__(self, name='roi_bbox_loss', **kwargs):
        super(Roi_bbox_loss, self).__init__(name='roi_bbox_loss', **kwargs)

    def call(self, inputs, **kwargs):
        """[summary]
        
        Arguments:
            delta_ {[type]} -- [n_rois, 4]
            delta {[type]} -- [n_rois, n_classes, 4]
            labels {[type]} -- [n_rois, 1]
            idxs_fg {[type]} -- [?, 1]
        """        
        delta_, delta, labels, idxs_fg = inputs
        
        n = tf.shape(idxs_fg)[0]
        idxs0 = tf.range(n)
        idxs1 = tf.gather(tf.cast(labels, tf.int32), idxs0)
        idxs = tf.stack([idxs0, idxs1[:,0]], axis=1)

        delta = tf.gather_nd(delta, idxs)
        delta_ = tf.gather(delta_, idxs0)
        return smooth_l1_loss(delta_, delta, sigma=1.)