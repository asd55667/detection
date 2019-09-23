
import os, time, torch
from darknet import Darknet53
from yolo_v3 import Yolo3
import config as cfg 

from utils import *
from generator import *


train_gen = OIDGen('split_1_annotation.json', batch_size=cfg.BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = Darknet53()
model = Yolo3(cfg.BATCH_SIZE, [cfg.ANCHOR_P3, cfg.ANCHOR_P4, cfg.ANCHOR_P5], 
                cfg.N_CLASSES, backbone).to(device)

model.apply(weights_init_normal)

optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    print(' ------------------- ' )
    model.train()
    start_time = time.time()
    for i, (imgs, targets, img_size) in enumerate(train_gen):
        print(' ------------------- ' )        
        imgs = totensor(imgs).to(device)
        imgs = imgs.permute(0,3,1,2).contiguous()
        
        targets = totensor(targets).to(device)
        targets.requires_grad = False

        outputs, loss = model(imgs, targets)
        print(loss.detach().cpu())
        loss.backward()

        batches_done = len(train_gen) * epoch + i
        if batches_done % 2:
            # Accumulates gradient before each step
            optimizer.step()
            optimizer.zero_grad()        
    train_gen.shuffle()