import numpy as np
import tensorflow as tf 

def generate_anchors(base_size, ratios, scales):
    base_anchor = np.array([0,0, base_size-1, base_size-1])
    x,y,w,h = xxyy2xywh(base_anchor)
    ws = np.round(np.sqrt(w * h / ratios))
    hs = np.round(ws * ratios)
    ratio_anchors = np.stack([x-(ws-1)/2, y-(hs-1)/2, x+(ws-1)/2, y+(hs-1)/2], axis=1)
    x,y,w,h = np.split(xxyy2xywh(ratio_anchors),4,1)
    ws = (w * scales).flatten()
    hs = (h * scales).flatten()
    x = np.repeat(x.flatten(),3)
    y = np.repeat(y.flatten(),3)
    anchors = np.stack([x-(ws-1)/2, y-(hs-1)/2, x+(ws-1)/2, y+(hs-1)/2], axis=1)
    return anchors

def xxyy2xywh(bbox):
    dim = bbox.ndim-1
    x0,y0,x1,y1 = np.split(bbox, 4, dim)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    x = x0 + (w-1) * 0.5
    y = y0 + (h-1) * 0.5
    return np.hstack([x,y,w,h])

    
def shift(anchors, stride, map_size):
    w, h = map_size[0], map_size[1]
    x = tf.range(w, ) * stride
    y = tf.range(h, ) * stride
    xx, yy = tf.meshgrid(x, y)
    xx = tf.reshape(xx,(-1,))
    yy = tf.reshape(yy,(-1,))
    shifts = tf.transpose(tf.stack([xx, yy, xx, yy]))

    anchors = tf.cast(anchors[None,], tf.float32)
    shifts = tf.cast(shifts[:,None], tf.float32)
    all_anchors = tf.reshape((anchors + shifts),(-1, 4))
    return all_anchors


def shift_np(anchors, stride, map_size):

    w, h = map_size
    x = np.arange(0, w) * stride
    y = np.arange(0, h) * stride
    xx, yy = np.meshgrid(x, y)
    shifts = np.stack([xx.ravel(), yy.ravel(), xx.ravel(), yy.ravel()]).T

    anchors = np.expand_dims(anchors, 0)
    shifts = np.expand_dims(shifts, 1)
    return (anchors + shifts).reshape((-1, 4))

def bbox_transform(anchors, gt_bbox, mean=None, std=None):
    if not mean: mean = np.array([0, 0, 0, 0])
    if not std: std = np.array([0.2, 0.2, 0.2, 0.2])
        
    w_a = anchors[:, 2] - anchors[:, 0] + 1
    h_a = anchors[:, 3] - anchors[:, 1] + 1
    x_a = anchors[:, 0] + w_a * 0.5
    y_a = anchors[:, 1] + h_a * 0.5

    w_gt = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
    h_gt = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
    x_gt = gt_bbox[:, 0] + w_gt * 0.5
    y_gt = gt_bbox[:, 1] + h_gt * 0.5

    dx = ((x_gt - x_a)/w_gt)[:, None]
    dy = ((y_gt - y_a)/h_gt)[:, None]
    dw = np.log(w_gt / w_a)[:, None]
    dh = np.log(h_gt / h_a)[:, None]

    return (np.hstack([dx, dy, dw, dh]) - mean ) / std

def bbox_transform_tf(rois, gt_bbox, mean=None, std=None):
    if not mean: mean = np.array([0, 0, 0, 0])
    if not std: std = np.array([0.2, 0.2, 0.2, 0.2])

    w_r = rois[:, 2] - rois[:, 0] + 1
    h_r = rois[:, 3] - rois[:, 1] + 1
    x_r = rois[:, 0] + (w_r) * .5
    y_r = rois[:, 1] + (h_r) * .5

    w_gt = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
    h_gt = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
    x_gt = gt_bbox[:, 0] + w_gt * 0.5
    y_gt = gt_bbox[:, 1] + h_gt * 0.5

    dx = ((x_gt - x_r) / w_gt)
    dy = ((y_gt - y_r) / h_gt)
    dw = tf.log(w_gt / w_r)
    dh = tf.log(h_gt / h_r)
    return (tf.stack([dx, dy, dw, dh], axis=1) - mean) / std




def bbox_transform_inv(anchors, delta):
    w = anchors[:, 2] - anchors[:, 0] + 1
    h = anchors[:, 3] - anchors[:, 1] + 1
    x = anchors[:, 0] + w * 0.5
    y = anchors[:, 1] + h * 0.5

    dx = delta[:, 0]
    dy = delta[:, 1]
    dw = delta[:, 2]
    dh = delta[:, 3]

    x += dx * w
    y += dy * h
    w *= tf.exp(dw)
    h *= tf.exp(dh)

    return tf.stack([x - w/2, y - h/2, x + w/2, y + h/2], axis=1) 


def compute_overlaps(bbox1, bbox2):
    # 这是在数像素的个数，相减的结果是它们的间距，再加一个端点就是总个数了  归一化后就不能+1了
    area1 = (bbox1[:, 2] - bbox1[:, 0] + 1) * \
        (bbox1[:, 3] - bbox1[:, 1] + 1)    # (M,)
    area2 = (bbox2[:, 2] - bbox2[:, 0] + 1) * \
        (bbox2[:, 3] - bbox2[:, 1] + 1)    # (N,)

    x1 = np.maximum(bbox1[:, 0, None], bbox2[:, 0])    # (M,N)
    y1 = np.maximum(bbox1[:, 1, None], bbox2[:, 1])
    x2 = np.minimum(bbox1[:, 2, None], bbox2[:, 2])
    y2 = np.minimum(bbox1[:, 3, None], bbox2[:, 3])
    w = np.maximum(x2 - x1 + 1, 0)
    h = np.maximum(y2 - y1 + 1, 0)
    intersect = w * h

    return intersect / (area1[:, None] + area2 - intersect)

def compute_overlaps_tf(bbox1, bbox2):

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