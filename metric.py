import numpy as np 

import torch 
from torch.nn import functional as F 

class Metrics:
    def __init__(self, predicted, original_mask, loss, eps = 1e-10, no_of_class = 2):
        
        self.pred, self.mask = torch.argmax(F.softmax(predicted, dim=1), dim=1), original_mask # batch, width, height
        self.loss = loss 
        self.eps = eps 
        self.no_of_class = no_of_class 
        self.pred_ = predicted 
        ...
    
    def to_contiguous(self, inp):
        return inp.contiguous().view(-1)
    
    def PA(self):
        with torch.no_grad():
            match  = torch.eq(self.pred, self.mask).int()

        return float(match.sum()) / float(match.numel())
    
    def mIoU(self):
        with torch.no_grad():
            self.mask = torch.argmax(self.mask, dim=1) 
            pred, mask = self.to_contiguous(self.pred), self.to_contiguous(self.mask)
            iou_per_class = [] 
            for c in range(self.no_of_class):
                match_pred = pred == c 
                match_mask = mask == c 

                if match_mask.long().sum().item() == 0:
                    iou_per_class.append(np.nan)
                else:
                    intersect = torch.logical_and(match_pred, match_mask).sum().float().item() 
                    union = torch.logical_or(match_pred, match_mask).sum().float().item() 

                    iou = (intersect + self.eps) /(union + self.eps) 
                    iou_per_class.append(iou)
            return np.nanmean(iou_per_class)
    
    def calculate_loss(self):
        return self.loss_fn(self.pred_, torch.argmax(self.mask, dim=1))