from keras.utils import Sequence
import json
import os

import numpy as np 
from PIL import Image

from cfg import cfg
from anchors import *

class OIDGen(Sequence):
    def __init__(self, json_file, batch_size=cfg.BATCH_SIZE, min_size=cfg.MIN_SIZE, max_size=cfg.MAX_SIZE):
        self.batch_size = batch_size
        self.max_size = max_size
        self.min_size = min_size

        with open(cfg.LEVEL_1_DIR + json_file, 'r') as f:
            self.annotations = json.load(f)
        ids = self.annotations.keys()
        self.idx2id = dict(zip(range(len(ids)), list(ids)))
        self.idx = list(range(len(self.idx2id)))
        self.on_epoch_end()

    def __len__(self,):
        return (len(self.idx) + self.batch_size -1) // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.idx)

    def __getitem__(self, idx):
        start = idx // self.batch_size
        end = min(start + self.batch_size, len(self.idx))

        imgs = []
        gt_bboxes = []
        labels = []
        scales = []
        for i in range(start, end):
            img, gt_bbox, label, scale, anchor_target = self.load_img(i)
            imgs.append(img)
            gt_bboxes.append(gt_bbox)
            labels.append(label)
            scales.append(scale)
        return img, gt_bbox, label, scale
    
    def img_path(self, idx):
        return self.annotations[self.idx2id[idx]]['p']

    def load_img(self, idx):
        idx = self.idx[idx]
        img_info = self.annotations[self.idx2id[idx]]
        w, h = img_info['w'], img_info['h']

        boxes = img_info['boxes']
        gt_bbox = []
        labels = []
        for box in boxes:
            info = list(box.values())
            labels.append(info[0])
            gt_bbox.append(info[1:])        
        
        gt_bbox = np.array(gt_bbox) * [w,h,w,h]

        scale = min(self.min_size / min(h,w), self.max_size / max(h,w))
        img = Image.open(self.img_path(idx)).resize((int(w * scale), int(h * scale)))
        img = np.expand_dims(np.array(img),0)
        
        gt_bbox *= scale
        gt_bbox = np.array(np.round(gt_bbox), dtype=np.int32)
        anchor_target = get_anchor_target(img.shape[1:3] ,gt_bbox, cfg)
        # after np.array(img) img.shape = (h, w, c)
        return img, gt_bbox, labels, scale, anchor_target


class ValidationGen(OIDGen):

    def __init__(self,data_dir=cfg.VALIDATION_DIR, json_file='validaiton_level_1.json'): 
        self.dir = data_dir   
        super(ValidationGen, self).__init__(json_file)
        

    def img_path(self, idx):
        filename = self.idx2id[idx] + '.jpg'
        return self.dir + filename
    
    

class TestGen(Sequence):
    def __init__(self, data_dir=cfg.TEST_DIR, batch_size=cfg.BATCH_SIZE, aug=cfg.MULTI_SCALE_TESTING):
        self.dir = data_dir
        self.batch_size = batch_size
        self.aug = aug
        self.pth = os.listdir(self.dir)

    def __len__(self,):
        return (len(self.pth) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx // self.batch_size
        end = (start + self.batch_size, len(self.pth))
        imgs = []
        for i in range(start, end):
            img = self.load_img(i)
            imgs.append(img)
        return imgs

    def load_img(self, idx):
        if self.aug:
        # TODO
            raise NotImplementedError
        else:
            h, w = np.array(Image.open(self.pth[idx])).shape[:2]            
            scale = min(cfg.MIN_SIZE / min(h,w), cfg.MAX_SIZE / max(h,w))
            img = Image.open(self.img_path(idx)).resize((int(w * scale), int(h * scale)))
            img = np.expand_dims(np.array(img),0)        
            return img, scale    
            


def get_anchor_target(img_size, gt_bbox, cfg):
    # generate delta_

    anchors = generate_anchors(cfg.BASE_SIZE, cfg.ANCHOR_RATIO, cfg.ANCHOR_SCALE)
    H, W = img_size
    map_size = [int(v / cfg.FEAT_STRIDE) for v in img_size]
    all_anchors = shift_np(anchors, cfg.FEAT_STRIDE, map_size)
    
    idxs_inside = np.where((all_anchors[:,0] >= 0) & 
                        (all_anchors[:,1] >= 0) &
                        (all_anchors[:,2] <= W) &
                        (all_anchors[:,3] <= H))[0]
    n = len(all_anchors)
    anchors = all_anchors[idxs_inside, :]    
    
    ious = compute_overlaps(np.ascontiguousarray(anchors, dtype=np.float32),
                            np.ascontiguousarray(gt_bbox, dtype=np.float32))

    argmax_anchors = ious.argmax(axis=1)
    argmax_gt = ious.argmax(axis=0)
    max_anchors = ious[np.arange(ious.shape[0]), argmax_anchors]
    max_gt = ious[argmax_gt, np.arange(ious.shape[1])]
    
    
    labels = -1 * np.ones((ious.shape[0]), dtype=np.float32)

    labels[max_anchors < cfg.NEG_THRESH] = 0
    labels[argmax_gt] = 1
    labels[max_anchors > cfg.POS_THRESH] = 1
    
    pos_idxs = np.where(labels == 1)[0]
    neg_idxs = np.where(labels == 0)[0]
    
    n_fg = int(cfg.RPN_BATCH_SIZE * cfg.FG_RATIO)
    if len(pos_idxs) > n_fg:
        disabled = np.random.choice(pos_idxs, size=len(pos_idxs) - n_fg, replace=False)
        labels[disabled] = -1

    n_bg = cfg.RPN_BATCH_SIZE - n_fg
    if len(neg_idxs) > n_bg:
        disabled = np.random.choice(neg_idxs, size=len(neg_idxs) - n_bg, replace=False)
        labels[disabled] = -1
    
    delta_ = bbox_transform(anchors, gt_bbox[argmax_anchors,:])
    
    def _unmap(data, rows, idxs, fill):
        if len(data.shape) == 1:
            all_data = np.empty((rows, ), dtype=np.float32)
            all_data.fill(fill)
            all_data[idxs,] = data
        else:
            all_data = np.empty((rows,) + data.shape[1:], dtype=np.float32)
            all_data.fill(fill)
            all_data[idxs,:] = data
        return all_data     

    anchor_labels = _unmap(labels, n, idxs_inside, -1)
    delta_ = _unmap(delta_, n, idxs_inside, 0)
    return delta_, anchor_labels 

