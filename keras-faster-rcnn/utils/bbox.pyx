import numpy as np
cimport cython
cimport numpy as np



def compute_overlaps(np.ndarray[np.float_t, ndim=2] bbox1,
                     np.ndarray[np.float_t, ndim=2] bbox2):
    cdef unsigned int m = bbox1.shape[0]
    cdef unsigned int n = bbox2.shape[0]
    cdef np.ndarray[np.float_t, ndim = 2] iou = np.zeros((m, n), dtype=np.float)
    cdef np.float_t area1, area2, w, h
    cdef unsigned int i, j

    for i in range(m):
        area1 = (bbox1[i, 2]-bbox1[i, 0]+1) * (bbox1[i, 3]-bbox1[i, 1]+1)

        for j in range(n):

            w = min(bbox1[i, 2], bbox2[j, 2]) - \
                max(bbox1[i, 0], bbox2[j, 0]) + 1
            if w > 0:
                h = min(bbox1[i, 3], bbox2[j, 3]) - \
                    max(bbox1[i, 1], bbox2[j, 0]) + 1
                if h > 0:
                    area2 = (bbox2[j, 2]-bbox2[j, 0] + 1) * \
                        (bbox2[j, 3]-bbox2[j, 1] + 1)
                    iou[i, j] = h*w / (area1+area2-h*w)
    return iou


def nms(np.ndarray[np.float_t, ndim=2] dets, np.float_t thresh):    
    # cdef np.ndarray[np.float_t, ndim=1] order np.argsort(dets[:,4])[::-1]
    cdef np.ndarray[np.float_t, ndim=1] score = dets[:,4]
    cdef np.ndarray[np.int_t, ndim=1] order = np.argsort(score)[::-1]
    
    cdef int m = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] supress = np.zeros((m), dtype=np.int)
    cdef np.float_t iou
    cdef int i, j, _i, _j

    keep = []
    for _i in range(m):
        i = order[_i]
        if supress[i] == 1:
            continue
        keep.append(i)

        for _j in range(_i+1,m):
            j = order[_j]
            if supress[j] == 1:
                continue 
            iou = compute_overlaps(dets[None, i, :4], dets[None, j, :4])
            if iou > thresh:
                supress[j] = 1
    return keep


def soft_nms(np.ndarray[np.float_t, ndim=2] dets, np.float_t thresh, 
             np.float_t sigma=0.75, np.float_t score_thresh=0.01, np.int_t method=0):
    
    cdef np.ndarray[np.float_t, ndim=1] score = dets[:,4]
    cdef np.ndarray[np.int_t, ndim=1] order = np.argsort(score)[::-1]
    
    cdef int m = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] supress = np.zeros((m), dtype=np.int)
    cdef np.float_t iou, w
    cdef int i, j, _i, _j

    keep = []
    for _i in range(m):
        i = order[_i]
        if supress[i] == 1:
            continue
        keep.append(i)

        for _j in range(_i+1,m):
            j = order[_j]
            if supress[j] == 1:
                continue 
            iou = compute_overlaps(dets[None, i, :4], dets[None, j, :4])[0]

            if iou > thresh:
                if method == 0:
                    w = np.exp(-1 *(1-iou)**2/sigma)
                elif method == 1:
                    w = 1 - iou
                elif method == 2:
                    w = 0
                score[j] = score[j] * w
                if score[j] < score_thresh:
                    supress[j] = 1

    return keep
            

