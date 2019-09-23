from darknet import dbl
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
from utils import *
import time, os  



def dbl5(in_channels, out_channels):
    l1 = [dbl(in_channels, out_channels, 1,1,0)]
    l2 = [dbl(out_channels, out_channels*2, 3,1,1)]
    l3 = [dbl(out_channels*2, out_channels, 1,1,0)]
    layers = l1 + l2 + l3 + l2 + l3
    return nn.Sequential(*layers)


class Upsample(nn.Module):
    def __init__(self, factor, mode='nearest'):
        super(Upsample, self).__init__()
        self.mode = mode 
        self.factor = factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Pyrmid(nn.Module):
    def __init__(self, in_channels, out_channels, not_last=True):       
        super(Pyrmid, self).__init__()
        self.dbl5 = dbl5(in_channels, out_channels)
        if not_last:
            self.dbl_u = dbl(out_channels, out_channels//2, 1, 1, 0)
            self.upsample = Upsample(2)

        self.dbl = dbl(out_channels, out_channels*2, 3,1,1)
        self.conv = nn.Conv2d(out_channels*2, 3*(5+443), 1, 1, 0)
        self.not_last = not_last
    
    def forward(self, x):
        _C = self.dbl5(x)
        P = self.dbl(_C)
        P = self.conv(P)

        if self.not_last:
            _P = self.dbl_u(_C)
            _P = self.upsample(_P)
        return [_P, P] if self.not_last else P

        

class Yolo3(nn.Module):
    def __init__(self, batch_size, anchors, num_classes, backbone):
        super(Yolo3, self).__init__()
        self.backbone = backbone
        self.anchors = anchors
        self.batch_size = batch_size
        self.p5 = Pyrmid(1024,512)
        self.p4 = Pyrmid(768, 256)
        self.p3 = Pyrmid(384, 128, None)

        self.num_classes = num_classes
        self.ignore_thres = 0.5
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.obj_scale = 1
        self.noobj_scale = 100

    def forward(self, x, targets=None):
        C3, C4, C5 = self.backbone(x)
        C4_, P5 = self.p5(C5)
        C3_, P4 = self.p4(torch.cat([C4, C4_],1))
        P3 = self.p3(torch.cat([C3, C3_], 1))

        outputs = []
        losses = []
        for i, pred in enumerate([P3, P4, P5]):
            grid_size = pred.shape[-1]
            pred = pred.view(self.batch_size, 3, self.num_classes+5, grid_size, grid_size).premute(0,1,3,4,2).contiguous()
            x = torch.sigmoid(pred[..., 0].float())
            y = torch.sigmoid(pred[..., 1].float())
            w = pred[..., 2]
            h = pred[..., 3]
            conf = torch.sigmoid(pred[..., 4].float())
            clss = torch.sigmoid(pred[..., 5:].float())
            
            delta = torch.cat([x[..., None], y[..., None], pred[..., 2:4]], axis=-1)
            bbox, anchors = bbox_transform(tonumpy(delta), self.anchors[i], grid_size)
            
            
            output = torch.cat(bbox.view(self.batch_size, -1, 4), 
                                        conf.view(self.batch_size,-1,1), 
                                        clss.view(self.batch_size,-1, self.num_classes),-1)

            if targets is None:
                continue
            else:
                iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
                    pred_boxes=bbox,
                    pred_cls=clss,
                    target=targets,
                    anchors=anchors,
                    ignore_thres=self.ignore_thres,
                )
                    
                loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
                loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
                loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
                loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
                loss_conf_obj = self.bce_loss(conf[obj_mask], tconf[obj_mask])
                loss_conf_noobj = self.bce_loss(conf[noobj_mask], tconf[noobj_mask])
                loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
                loss_cls = self.bce_loss(clss[obj_mask], tcls[obj_mask])
                loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls                


                cls_acc = 100 * class_mask[obj_mask].mean()
                conf_obj = conf[obj_mask].mean()
                conf_noobj = conf[noobj_mask].mean()
                conf50 = (conf > 0.5).float()
                iou50 = (iou_scores > 0.5).float()
                iou75 = (iou_scores > 0.75).float()
                detected_mask = conf50 * class_mask * tconf
                precision = torch.sum(iou50 * detected_mask) / (conf50.sum() + 1e-16)
                recall50 = torch.sum(iou50 * detected_mask) / (obj_mask.sum() + 1e-16)
                recall75 = torch.sum(iou75 * detected_mask) / (obj_mask.sum() + 1e-16)

                metric = {
                    "loss": to_cpu(loss).item(),
                    "x": to_cpu(loss_x).item(),
                    "y": to_cpu(loss_y).item(),
                    "w": to_cpu(loss_w).item(),
                    "h": to_cpu(loss_h).item(),
                    "conf": to_cpu(loss_conf).item(),
                    "cls": to_cpu(loss_cls).item(),
                    "cls_acc": to_cpu(cls_acc).item(),
                    "recall50": to_cpu(recall50).item(),
                    "recall75": to_cpu(recall75).item(),
                    "precision": to_cpu(precision).item(),
                    "conf_obj": to_cpu(conf_obj).item(),
                    "conf_noobj": to_cpu(conf_noobj).item(),
                    "grid_size": grid_size,
                }
                losses.append(loss)
                outputs.append(output)
        total_loss = sum(losses)
        return outputs, total_loss

    def save(self, pth):
        save_dict = dict()
        save_dict['model'] = self.backbone.state_dict()
        
        save_dir = os.path.dirname(pth)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)     
        torch.save(save_dict, pth)
           

def to_cpu(tensor):
    return tensor.detach().cpu()



    # def load_weights(self, pth):
    #     with open(pth, 'r') as f:
    #         header = np.fromfile(f, dtype=np.int32, count=5)
    #         self._head_info = header
    #         self._seen = header[3]
    #         weights = np.fromfile(f, dtype=np.float32)
    #     cutoff = None




if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Yolo3(np.random.rand(10,4)).to(device)
    summary(model, (3, 416, 416))