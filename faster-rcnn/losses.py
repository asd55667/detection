import tensorflow as tf 
from keras import layers


def smooth_l1_loss(delta_, delta, sigma=3.):        
    x = tf.abs(delta - delta_)
    cond = tf.less(x, 1./sigma**2)
    loss = tf.where(cond, .5*sigma**2 * tf.math.pow(x,2), x - .5 / sigma**2)
    return tf.reduce_sum(loss)


def rpn_class_loss(y_, y):
    # y_ = tf.cast(tf.equal(y_[:,4], 1), tf.int32)
    y_ = y_[:,4]
    idxs = tf.where(tf.not_equal(y_, -1))
    n = tf.shape(idxs)[0]

    y_ = tf.gather_nd(y_, idxs)

    y = y[0,:, 1]
    y = tf.gather_nd(y, idxs)    

    return tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y) / tf.cast(n, tf.float32)



def rpn_bbox_loss(delta_label, delta):
    delta_ = delta_label[:,:4]
    labels = delta_label[:, 4]

    idxs = tf.where(tf.equal(labels, 1))
    delta = tf.gather_nd(delta, idxs)
    n = tf.shape(idxs)[0]
    return smooth_l1_loss(delta_, delta) / tf.cast(n, tf.float32)


def roi_class_loss(y_, y):
    """[summary]
    
    Arguments:
        y_ {[type]} -- [n_rois, 1]
        y {[type]} -- [batch_size=1, n_rois, n_classes]    
    """
    
    n = tf.shape(y_)[0]
    y = y[0]
    y_ = tf.cast(y_[:,0], tf.int32)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y) / tf.cast(n, tf.float32)

class Roi_class_loss(layers.Layer):
    def __init__(self, name='roi_class_loss', **kwargs):
        super(Roi_class_loss, self).__init__(name='roi_class_loss', **kwargs)

    def call(self, inputs, **kwargs):
        y_, y = inputs
        n = tf.shape(y_)[0]
        y = y[0]
        y_ = tf.cast(y_[:,0], tf.int32)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y) / tf.cast(n, tf.float32)        



def roi_bbox_loss(delta_, delta, labels, idxs_fg):
    """[summary]
    
    Arguments:
        delta_ {[type]} -- [n_rois, 4]
        delta {[type]} -- [batch_size=1, n_rois, n_classes, 4]
        labels {[type]} -- [n_rois, 1]
        idxs_fg {[type]} -- [?, 1]
    """

    delta = delta[0]

    n = tf.shape(idxs_fg)[0]
    idxs0 = tf.range(n)
    idxs1 = tf.gather(tf.cast(labels, tf.int32), idxs0)
    idxs = tf.stack([idxs0, idxs1[:,0]], axis=1)

    delta = tf.gather_nd(delta, idxs)
    delta_ = tf.gather(delta_, idxs0)
    return smooth_l1_loss(delta_, delta, sigma=1.) / tf.cast(n, tf.float32)
