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

    return (tf.stack([dx, dy, dw, dh],axis=1) - mean ) / std


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
