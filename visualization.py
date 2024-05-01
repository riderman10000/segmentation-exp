import cv2 
import random 
import numpy as np 
import matplotlib.pyplot as plt 
from torchvision import transforms 

def tensor_2_numpy(data, mask = None):
    inverse_transform = transforms.Compose([
        transforms.Normalize(
            mean=[0., 0., 0.],
            std=[1/0.229, 1/0.224, 1/0.225],
        ),
        transforms.Normalize(
            mean = [ -0.485, -0.456, -0.406 ],
            std = [ 1., 1., 1. ],
        )
    ])
    # rgb = True if len(data) == 3 else False
    # return (inverse_transform(data) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if rgb else (data*255).detach().cpu().numpy().astype(np.uint8)
    rgb = True if not mask else False
    img = ((data) * 255).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8) if rgb else (data*255).detach().cpu().numpy().astype(np.uint8)
    
    
    return img 

def visualize(dataset, no_of_images):
    plt.figure(figsize=(25, 20)) 
    rows = no_of_images // 4 
    columns = no_of_images//rows

    indices = [random.randint(0, len(dataset) - 1) for _ in range(no_of_images//2)]
    count = 1
    for i, index in enumerate(indices):
        print(i, count)
        image, mask = dataset[index] 
        # image 
        plt.subplot(rows, columns, count)
        plt.imshow(tensor_2_numpy(image.squeeze(0).float()))
        count += 1
        # mask
        plt.subplot(rows, columns, count)
        plt.imshow(tensor_2_numpy(mask.squeeze(0).float()))
        count += 1 
    plt.show()