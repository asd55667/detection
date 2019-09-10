import numpy as np


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


def soft_nms(dets,  iou_thresh=0.5 ,score_thresh=0.1, sigma=0.75, method='guassian'):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    area = (x2-x1+1)*(y2-y1+1)
    score = dets[:,4]
    order = np.argsort(score)[::-1]

    keep = []
    
    while order.size > 0:
        i = order[0]
        # j += 1
        # if i in keep:
        #     continue
        
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(xx2-xx1+1, 0)
        h = np.maximum(yy2-yy1+1, 0)
        intersect = w * h
        iou = intersect / (area[i] + area[order[1:]] - intersect)
        
        # idxs1 sorted level 
        # idxs2 idxs3 origin level   
        idxs1 = np.where(iou > iou_thresh)[0]
        idxs2 = order[idxs1 + 1]
        
        if method == 'guassian':
            W = np.exp(-1*iou[idxs1]**2/sigma)
        elif method == 'linear':
            W = 1 - iou[idxs1]
        elif method == 'nms':
            W = np.zeros((idxs2.shape[0],))
        score[idxs2] *= W
        score[i] = 0
        idxs3 = np.where(score>=score_thresh)[0]
        order = score.argsort()[::-1][:len(idxs3)]
    
    return keep

# from bbox import *
a=np.array([[100,100,210,210,0.72],
        [250,250,420,420,0.8],
        [220,220,320,330,0.92],
        [100,100,210,210,0.72],
        [230,240,325,330,0.81],
        [220,230,315,340,0.9]])


print(soft_nms(a, 0.5, method='nms'))

# print(compute_overlaps())
