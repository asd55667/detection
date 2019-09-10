import numpy as np
import tensorflow as tf 

def generate_anchors(base_size=16, scale=2**np.arange(3,6), ratio=[0.5, 1, 2]):
    """[summary]
    
    Keyword Arguments:
        base_size {int} -- [description] (default: {16})
        scale {[type]} -- [description] (default: {2**np.arange(3,6)})
        ratio {list} -- [description] (default: {[0.5, 1, 2]})
    
    Returns:
        [nparray] -- [(9,4)]
    """
    x1 = [base_size * s * r for s in scale for r in ratio]
    y1 = [base_size * s for _ in range(len(ratio)) for s in scale]
    x1y1 = np.vstack([x1, y1]).T

    x0y0 = np.zeros((len(scale)*len(ratio), 2))
    bboxes = np.hstack([x0y0, x1y1])

    noise = np.random.randint(-base_size, base_size,
                              size=(len(scale)*len(ratio), 2))

    bboxes[:, 0::4] = -bboxes[:, 2::4]/2 + noise[:, 0::2]
    bboxes[:, 2::4] = bboxes[:, 2::4]/2 + noise[:, 0::2]
    bboxes[:, 1::4] = -bboxes[:, 3::4]/2 + noise[:, 1::2]
    bboxes[:, 3::4] = bboxes[:, 3::4]/2 + noise[:, 1::2]

    return tf.Variable(bboxes,dtype=tf.float32)

# def generate_anchors(base_size=16, ratios, scales=None):
#     n = len(ratios) * len(scales)

#     anchors = np.zeros((n, 4))

#     anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

#     areas = anchors[:, 2] * anchors[:, 3]

#     anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
#     anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

#     anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
#     anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

#     return anchors    


def bbox_transform_inv(anchors, delta):
    """[summary]
    
    Arguments:
        anchors {[ndarray]} -- [(n,4)]
        delta {[tensor]} -- [(n,4)]
    
    Returns:
        [tensor] -- [(n,4)]
    """
    w = anchors[:, 2] - anchors[:, 0] + 1
    h = anchors[:, 3] - anchors[:, 1] + 1
    x_ctr = anchors[:, 0] + w * 0.5
    y_ctr = anchors[:, 1] + h * 0.5

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    x_ctr += dx * w
    y_ctr += dy * h
    w *= tf.exp(dw)
    h *= tf.exp(dh)

    return tf.stack([x_ctr - w/2, y_ctr - h/2, x_ctr + w/2, y_ctr + h/2], axis=1)


def shift(anchors, stride, map_size):
    """[summary]
    
    Arguments:
        anchors {[tensor]} -- [description]
        stride {[type]} -- [description]
        map_size {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    w, h = map_size[0], map_size[1]
    x = tf.range(w, dtype=tf.float32) * stride
    y = tf.range(h, dtype=tf.float32) * stride
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(xx,(-1,))
    yy = tf.reshape(yy,(-1,))
    shifts = tf.transpose(tf.stack([xx, yy, xx, yy]))

    anchors = tf.expand_dims(anchors, 0)
    shifts = tf.expand_dims(shifts, 1)
    all_anchors = tf.reshape((anchors + shifts),(-1, 4))
    return all_anchors


def bbox_transform(anchors, gt_bbox):
    """[summary]
    compute delta between anchors and gt_bbox
    Arguments:
        anchors {[tensor]} -- [description]
        gt_bbox {[ndarray]} -- [description]
    
    Returns:
        [tensor] -- [description]
    """
    w_a = anchors[:, 2] - anchors[:, 0] + 1
    h_a = anchors[:, 3] - anchors[:, 1] + 1
    x_a_ctr = anchors[:, 0] + w_a * 0.5
    y_a_ctr = anchors[:, 1] + h_a * 0.5

    w_gt = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
    h_gt = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
    x_gt_ctr = gt_bbox[:, 0] + w_gt * 0.5
    y_gt_ctr = gt_bbox[:, 1] + h_gt * 0.5

    dx = ((x_gt_ctr - x_a_ctr)/w_gt)
    dy = ((y_gt_ctr - y_a_ctr)/h_gt)
    dw = tf.log(w_gt / w_a)
    dh = tf.log(h_gt / h_a)
    delta_ = tf.stack([dx, dy, dw, dh], axis=1)
    return delta


def clip_bbox(boxes, img_size):

    xlim, ylim = img_size
    x0 = tf.clip_by_value(boxes[:, 0], 0, xlim)
    y0 = tf.clip_by_value(boxes[:, 1], 0, ylim)
    x1 = tf.clip_by_value(boxes[:, 2], 0, xlim)
    y1 = tf.clip_by_value(boxes[:, 3], 0, ylim)

    return tf.stack([x0, y0, x1, y1],axis=1)

def filter_bboxs(boxes, min_size):
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return tf.reshape(tf.where(tf.logical_and(w>=min_size,h>=min_size)),[-1])


def compute_overlaps(bbox1, bbox2):

    w1 = bbox1[:,2] - bbox1[:,0] + 1
    h1 = bbox1[:,3] - bbox1[:,1] + 1
    area1 = w1 * h1
    
    w2 = bbox2[:,2] - bbox2[:,0] + 1
    h2 = bbox2[:,3] - bbox2[:,1] + 1
    area2 = w2 * h2

    x1 = tf.maximum(bbox1[:, 0, None], bbox2[:, 0])    # (M,N)
    y1 = tf.maximum(bbox1[:, 1, None], bbox2[:, 1])
    x2 = tf.minimum(bbox1[:, 2, None], bbox2[:, 2])
    y2 = tf.minimum(bbox1[:, 3, None], bbox2[:, 3])
    w = tf.maximum(x2 - x1 + 1, 0)
    h = tf.maximum(y2 - y1 + 1, 0)
    intersect = w * h

    return intersect / (area1[:, None] + area2 - intersect)


                           

