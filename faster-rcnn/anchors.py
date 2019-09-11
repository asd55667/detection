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
    return np.stack([x-(ws-1)/2, y-(hs-1)/2, x+(ws-1)/2, y+(hs-1)/2], axis=1)

def xxyy2xywh(bbox):
    dim = bbox.ndim-1
    x0,y0,x1,y1 = np.split(bbox, 4, dim)
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    x = x0 + (w-1) * 0.5
    y = y0 + (h-1) * 0.5
    return np.stack([x,y,w,h], axis=1)

    

def shift(anchors, stride, map_size):
    w,h = map_size
    x = (np.arange(0,w) + 0.5) * stride
    y = (np.arange(0,h) + 0.5) * stride
    xx, yy = np.meshgrid(x,y)
    shifts = np.stack([xx.ravel(), yy.ravel(), xx.ravel(), yy.ravel()]).T
    shifts = shifts[:,None]
    anchors = anchors[None,]
    return (anchors+shifts).reshape((-1,4))

def bbox_transform(anchors, gt_bbox, mean=None, std=None):
    if not mean: mean = np.array([0, 0, 0, 0])
    if not std: std = np.array([0.2, 0.2, 0.2, 0.2])
        
    w_a = anchors[:, 2] - anchors[:, 0] + 1
    h_a = anchors[:, 3] - anchors[:, 1] + 1
    x_a_ctr = anchors[:, 0] + w_a * 0.5
    y_a_ctr = anchors[:, 1] + h_a * 0.5

    w_gt = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
    h_gt = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
    x_gt_ctr = gt_bbox[:, 0] + w_gt * 0.5
    y_gt_ctr = gt_bbox[:, 1] + h_gt * 0.5

    dx = ((x_gt_ctr - x_a_ctr)/w_gt)[:, None]
    dy = ((y_gt_ctr - y_a_ctr)/h_gt)[:, None]
    dw = np.log(w_gt / w_a)[:, None]
    dh = np.log(h_gt / h_a)[:, None]

    return (np.hstack([dx, dy, dw, dh]) - mean ) / std


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