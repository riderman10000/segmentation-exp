import cv2 
import os 
import numpy as np 
from copy import deepcopy

import albumentations as A 
from albumentations.pytorch import ToTensorV2

import torch  
import torch.utils
import torch.utils.data
from torchvision import transforms 
from torch.nn import functional as F 
from torch.utils.data import DataLoader

from model import SegmentationModel
from football_dataset import FootballDataset

class_color_mapping = {
    0 : (137, 126, 126), # ground
    1 : (27,  71, 151), # advertisement
    2 : (111,  48, 253), # audience
    3 : (255,   0, 29), # football post
    4 : (255, 160, 1), # team A # orange | goal keeper in green of MU
    5 : (255, 159, 0), #  goal keeper in green of MU keeper
    6 : (254, 233, 3), # team B # yellow | goal keeper in yellow of RMA
    7 : (255, 235, 0), #  goal keeper in yellow of RMA
    8 : (238, 171, 171), # refree pink
    9 : (201,  19, 223), # football
}

image_size = 512 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
iteration_path = './runs/exp-c-optim/iteration-8'

video_capture = cv2.VideoCapture('./video/game1.mp4')
if not (video_capture.isOpened()):
    print('Error Openeing video stream or file')
# video_out = cv2.VideoWriter(
#     os.path.join(iteration_path, 'output.avi'),
#     cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))

video_out = cv2.VideoWriter(
    os.path.join(iteration_path, 'output1.avi'), 
    cv2.VideoWriter_fourcc(*"MJPG"), 20,(640,480))

segmentation_model = SegmentationModel()
segmentation_model.load_state_dict(torch.load(os.path.join(iteration_path, 'seg.pt')))
segmentation_model.to(device = device)


# albumentation_letterbox =
def letterbox(image, **kwargs):
    resized_width, resized_height = kwargs['resized_width'][0], kwargs['resized_height'][0]
    height, width, _ = image.shape
    scale = min(resized_height / height, resized_width / width)
    new_height = int(height * scale)
    new_width = int(width * scale)
    image = cv2.resize(image, (new_width, new_height))
    new_img = np.full((resized_height, resized_width, 3), 0,dtype='uint8')
    # fill new image with the resized image and centered it
    new_img[(resized_height - new_height) // 2:(resized_height - new_height) // 2 + new_height,
            (resized_width - new_width) // 2:(resized_width - new_width) // 2 + new_width,
            :] = image.copy()
    return new_img 
 
def custom_transform(data, **kwargs):
    kwargs['resized_width'] = image_size,
    kwargs['resized_height'] = image_size,
    return letterbox(image=data, **kwargs)

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if ret:
        transformations = A.Compose([
            A.Resize(image_size, image_size),
            # A.Lambda(name='Letter Box', 
            #     image=custom_transform,
            #     mask=custom_transform,
            #     p=1),
            
            ToTensorV2(),
        ])
        original_frame = deepcopy(frame)
        transformed_image = transformations(image=frame[:,:,::-1])['image']/255.0
        transformed_image = torch.tensor(transformed_image).unsqueeze(0)
        with torch.inference_mode():
            segmentation_model.eval()
            transformed_image = transformed_image.to(device)
            
            prediction = segmentation_model(transformed_image)
            segmented_mask = torch.argmax(F.softmax(prediction, dim=1), dim=1)
            segmented_mask = segmented_mask.permute(1, 2, 0).detach().cpu().numpy()
            _h, _w, _c = segmented_mask.shape
            segmented_image = np.zeros((_h, _w, 3))
            for row in range(_h):
                for col in range(_w):
                    segmented_image[row, col] = np.array(class_color_mapping[segmented_mask[row, col, 0]])
            segmented_image = segmented_image.astype(np.uint8)

            cv2.imshow('game', segmented_image[:,:,::-1])
            cv2.imshow('test', cv2.resize(original_frame, (image_size, image_size)))
            video_out.write(cv2.resize(segmented_image[:,:,::-1], (640, 480)).astype(np.uint8))
            key = cv2.waitKey(25)
            if key == ord('q'):
                cv2.destroyAllWindows()
                break 
    else:
        break
video_capture.release()
video_out.release()