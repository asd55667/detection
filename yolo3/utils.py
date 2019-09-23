import numpy as np
import torch as t

def shift(anchors, map_size, feat_stride):
    """[summary]
    
    Arguments:
        anchors {[type]} -- [3,4]
        map_size {[type]} -- []
        feat_stride {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    h, w = map_size
    x = np.arange(0, w) * feat_stride
    y = np.arange(0, h) * feat_stride
    xx, yy = np.meshgrid(x,y)
    shifts = np.stack([xx.ravel(), yy.ravel(), xx.ravel(), yy.ravel()], axis=0).T

    anchors = anchors[None,0]
    shifts = shifts[:,None,]
    return (anchors+shifts).reshaep((-1,4))

def bbox_transform(delta, anchors, grid_size,):
    """[summary]
    
    Arguments:
        delta {[type]} -- [batch_size, n_anchors, gride_size, gride_size, 4]
        anchors {[type]} -- [3,2] or list of 3 tuple
        grid_size {[type]} -- [13, 26, 52]
    """
    feat_stride = 416 // grid_size
    grid_x = np.tile(np.arange(grid_size), (grid_size, 1))[None, None,]
    grid_y = np.tile(np.arange(grid_size), (grid_size, 1)).T[None, None,]
    if type(anchors) != np.ndarray: 
        anchors = np.array(anchors)
    anchors /= feat_stride

    w_a = anchors[:, 0::2]
    h_a = anchors[:, 1::2]
    w_a.shape = h_a.shape = (1, anchors.shape[0], 1, 1)

    x, y, w, h = np.split(delta, 4, -1)
    x = x[...,0] + grid_x
    y = y[...,0] + grid_y
    w = np.exp(w[..., 0]) * w_a 
    h = np.exp(h[..., 0]) * h_a
    return np.stack([x, y, h, w], -1) * feat_stride, anchors





def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
    if isinstance(data, t.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor




def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = t.min(w1, w2) * t.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area  

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = t.max(b1_x1, b2_x1)
    inter_rect_y1 = t.max(b1_y1, b2_y1)
    inter_rect_x2 = t.min(b1_x2, b2_x2)
    inter_rect_y2 = t.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = t.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * t.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):
    """[summary]
    
    Arguments:
        pred_boxes {[type]} -- [batch_size, n_anchors, 13, 13,4]
        pred_cls {[type]} -- [batch_size, n_anchors, 13, 13, n_classes]
        target {[type]} -- [batch_size, 6]
        anchors {[type]} -- [3, 2]
        ignore_thres {[type]} -- [1]
    
    Returns:
        [type] -- [description]
    """
    ByteTensor = t.cuda.ByteTensor if pred_boxes.is_cuda else t.ByteTensor
    FloatTensor = t.cuda.FloatTensor if pred_boxes.is_cuda else t.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = t.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = t.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = t.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        t.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        t.nn.init.normal_(m.weight.data, 1.0, 0.02)
        t.nn.init.constant_(m.bias.data, 0.0)    


