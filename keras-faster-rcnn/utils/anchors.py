from utils import *
import numpy as np

def generate_anchors(base_size=16, scale=2**np.arange(3,6), ratio=[0.5, 1, 2]):

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

    return bboxes

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

    w = anchors[:, 2] - anchors[:, 0] + 1
    h = anchors[:, 3] - anchors[:, 1] + 1
    x_ctr = anchors[:, 0] + w * 0.5
    y_ctr = anchors[:, 1] + h * 0.5

    w = w[:, np.newaxis]
    h = h[:, np.newaxis]
    x_ctr = x_ctr[:, np.newaxis]
    y_ctr = y_ctr[:, np.newaxis]

    dx = delta[:, 0::4]
    dy = delta[:, 1::4]
    dw = delta[:, 2::4]
    dh = delta[:, 3::4]

    x_ctr += dx * w
    y_ctr += dy * h
    w *= np.exp(dw)
    h *= np.exp(dh)

    return np.hstack([x_ctr - w/2, y_ctr - h/2, x_ctr + w/2, y_ctr + h/2])


def shift(anchors, stride, map_size):

    w, h = map_size
    x = np.arange(0, w) * stride
    y = np.arange(0, h) * stride
    xx, yy = np.meshgrid(x, y)
    shifts = np.stack([xx.ravel(), yy.ravel(), xx.ravel(), yy.ravel()]).T

    anchors = np.expand_dims(anchors, 0)
    shifts = np.expand_dims(shifts, 1)
    return (anchors + shifts).reshape((-1, 4))


def bbox_transform(anchors, gt_bbox):

    w_a = anchors[:, 2] - anchors[:, 0] + 1
    h_a = anchors[:, 3] - anchors[:, 1] + 1
    x_a_ctr = anchors[:, 0] + w_a * 0.5
    y_a_ctr = anchors[:, 1] + h_a * 0.5

    w_gt = gt_bbox[:, 2] - gt_bbox[:, 0] + 1
    h_gt = gt_bbox[:, 3] - gt_bbox[:, 1] + 1
    x_gt_ctr = gt_bbox[:, 0] + w_gt * 0.5
    y_gt_ctr = gt_bbox[:, 1] + h_gt * 0.5

    dx = ((x_gt_ctr - x_a_ctr)/w_gt)[:, np.newaxis]
    dy = ((y_gt_ctr - y_a_ctr)/h_gt)[:, np.newaxis]
    dw = np.log(w_gt / w_a)[:, np.newaxis]
    dh = np.log(h_gt / h_a)[:, np.newaxis]

    return np.hstack([dx, dy, dw, dh])


def clip_bbox(boxes, img_size):

    xlim, ylim = img_size
    x0 = np.clip(boxes[:, 0::4], 0, xlim)
    y0 = np.clip(boxes[:, 1::4], 0, ylim)
    x1 = np.clip(boxes[:, 2::4], 0, xlim)
    y1 = np.clip(boxes[:, 3::4], 0, ylim)

    return np.hstack([x0, y0, x1, y1])

def filter_bboxs(boxes, min_size):
    w = boxes[:, 2::4] - boxes[:, 0::4]
    h = boxes[:, 3::4] - boxes[:, 1::4]
    return np.where(w <= min_size and h <= min_size)[0]



                           

