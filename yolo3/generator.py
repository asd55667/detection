import numpy as np 
import os, json
from PIL import Image 
from config import *


class OIDGen:
    def __init__(self, json_file, batch_size):
        self.batch_size = batch_size
        with open(LEVEL_1_DIR + json_file, 'r') as f:
            self.annotations = json.load(f)
        ids = self.annotations.keys()
        self.idx2id = dict(zip(range(len(ids)), list(ids)))
        self.idx = list(range(len(self.idx2id)))        
    
    def __getitem__(self, idx):
        start = idx // self.batch_size
        end = min(start + self.batch_size, len(self.idx))

        imgs = []
        gt_bboxes = []
        img_size = []        
        for i in range(start, end):
            img, gt_bbox, size = self.load_img(i)
            imgs.append(img)
            img_size.append(size)                        

            batch_idx = np.ones((gt_bbox.shape[0],1)) * i
            gt_bbox = np.hstack([batch_idx, gt_bbox])
            gt_bboxes.append(gt_bbox)
        imgs = np.concatenate(imgs, 0)
        gt_bboxes = np.concatenate(gt_bboxes, 0)
        return imgs, gt_bboxes, img_size

    def shuffle(self ):
        np.random.shuffle(self.idx)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __repr__(self):
        type_name = type(self).__name__
        imgs, gt_bbox, img_size = self[0]
        _format = '{} \n elements {} \n img_shape {!r} \n gt_shape {!r}, raw_size {!r}'
        return _format.format(type_name, len(self[0]), imgs.shape[1:], (None,gt_bbox.shape[1]), (None, None))


    def __len__(self,):
        return (len(self.idx) + self.batch_size -1) // self.batch_size

    def img_path(self, idx):
        return self.annotations[self.idx2id[idx]]['p']

    def load_img(self, idx):
        idx = self.idx[idx]
        img_info = self.annotations[self.idx2id[idx]]
        img_size = img_info['w'], img_info['h']

        boxes = img_info['boxes']
        gt_bbox = []
        for box in boxes:
            info = list(box.values())            
            gt_bbox.append(info)                
        gt_bbox = np.array(gt_bbox)

        img = Image.open(self.img_path(idx)).resize((416,416))
        img = np.array(img)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] > 3:
            img = img[:,:,:3]        
        img = img[None, ]
        return img, gt_bbox, img_size




a = OIDGen('split_1_annotation.json', 2)
