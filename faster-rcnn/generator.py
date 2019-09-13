from keras.utils import Sequence
import json
import os

import numpy as np 
from PIL import Image

from cfg import cfg


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
            img, gt_bbox, label, scale = self.load_img(i)
            imgs.append(img)
            gt_bboxes.append(gt_bbox)
            labels.append(label)
            scales.append(scale)
        return imgs, gt_bboxes, labels, scales
    
    def img_path(self, idx):
        return self.annotations[self.idx2id[idx]]['p']

    def load_img(self, idx):
        idx = self.idx[idx]
        img_info = self.annotations[self.idx2id[idx]]
        w, h = img_info['w'], img_info['h']
        scale = min(self.min_size / min(h,w), self.max_size / max(h,w))
        img = Image.open(self.img_path(idx)).resize((int(w * scale), int(h * scale)))
        img = np.expand_dims(np.array(img),0)
        boxes = img_info['boxes']
        gt_bbox = []
        labels = []
        for box in boxes:
            info = list(box.values())
            labels.append(info[0])
            gt_bbox.append(info[1:])
        gt_bbox = np.array(gt_bbox) * scale
        # after np.array(img) img.shape = (h, w, c)
        return img, gt_bbox, labels, scale



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
            return np.array(Image.open(self.pth[idx]))