import segmentation_models_pytorch as smp 
import torch 
from torch import nn 

class SegmentationModel(nn.Module):
    def __init__(self, encoder = 'timm-efficientnet-b0', weights = 'imagenet') -> None:
        super(SegmentationModel, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder, 
            encoder_weights=weights, 
            in_channels=3, 
            classes=3
        )
        # self.loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
        
    def forward(self, image, mask = None):
        pred = self.model(image) 
        if mask != None:  
            # print('argmax', torch.argmax(mask, dim=1))
            # print('mask', mask)

            # print('shape', torch.argmax(mask, dim=1).shape, mask.shape)
            # loss =  nn.CrossEntropyLoss()(pred, torch.argmax(mask, dim=1)) 
            loss = nn.BCEWithLogitsLoss()(pred, mask)
            return pred, loss 
        return pred 