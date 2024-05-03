import os 
import cv2
import numpy as np 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import albumentations as A 
from albumentations.pytorch import ToTensorV2

import torch  
import torch.utils
import torch.utils.data
from torchvision import transforms 
from torch.utils.data import DataLoader

from model import SegmentationModel
from football_dataset import FootballDataset

def train(epoch, model, batch_size, image_size, data_loader, optimizer,  device = 'cuda' if torch.cuda.is_available() else 'cpu', save_dirs = './runs/exp/'):
    model.train()

    train_loss = 0.0
    train_accuracy = 0.0 

    TQDM_BAR_FORMAT = "{l_bar}{bar:10}| {n_fmt}/{total_fmt} {elapsed}"  # tqdm bar format
    print(len(data_loader))

    # for i, (images, masks) in tqdm(enumerate(data_loader), total= len(data_loader.dataset), bar_format=TQDM_BAR_FORMAT):
    for images, masks in tqdm(data_loader): #, total= len(data_loader.dataset), bar_format=TQDM_BAR_FORMAT):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        pred, loss = model(images, masks)
        
        loss.backward()
        optimizer.step()

        train_loss += loss 

        accuracy = ((torch.sum(masks-pred).item())/(batch_size* 3 *image_size*image_size))*100
        train_accuracy += (100 - accuracy)

    if (epoch + 1) %10 ==0 :
        path_to_fig = os.path.join(save_dirs, 'train_pred')
        os.makedirs( path_to_fig, exist_ok=True)

        plt.subplot(1,3,1)
        plt.imshow(images[1].permute(1,2,0).detach().cpu().numpy())
        plt.axis('off')
        plt.title("Image")

        plt.subplot(1,3,2)
        plt.imshow(pred[1].permute(1,2,0).detach().cpu().numpy())
        plt.axis('off')
        plt.title("Prediction")

        plt.subplot(1,3,3)
        plt.imshow(masks[1].permute(1,2,0).detach().cpu().numpy())
        plt.axis('off')
        plt.title("Truth")

        plt.savefig(os.path.join(path_to_fig, f'image_{epoch+1}.jpg'))
        # plt.show()
    return train_loss / len(data_loader), train_accuracy/len(data_loader)

def valid(epoch, model: SegmentationModel, batch_size, image_size, data_loader, optimizer,  device = 'cuda' if torch.cuda.is_available() else 'cpu', save_dirs = './runs/exp/'):
    model.eval()
    with torch.inference_mode():
        infer_loss = 0.0
        infer_accuracy = 0.0 

        for images, masks in tqdm(data_loader): #, total= len(data_loader.dataset), bar_format=TQDM_BAR_FORMAT):
            images = images.to(device)
            masks = masks.to(device)

            pred, loss = model(images, masks)
            
            infer_loss += loss 

            accuracy = ((torch.sum(masks-pred).item())/(batch_size* 3 *image_size*image_size))*100
            infer_accuracy += (100 - accuracy)
        
        if (epoch+1) %10 ==0 :
            path_to_fig = os.path.join(save_dirs, 'train_pred')
            os.makedirs( path_to_fig, exist_ok=True)

            plt.subplot(1,3,1)
            plt.imshow(images[1].permute(1,2,0).detach().cpu().numpy())
            plt.axis('off')
            plt.title("Image")

            plt.subplot(1,3,2)
            plt.imshow(pred[1].permute(1,2,0).detach().cpu().numpy())
            plt.axis('off')
            plt.title("Prediction")

            plt.subplot(1,3,3)
            plt.imshow(masks[1].permute(1,2,0).detach().cpu().numpy())
            plt.axis('off')
            plt.title("Truth")

            plt.savefig(os.path.join(path_to_fig, f'image_{epoch+1}.jpg'))
            # plt.show()
        return infer_loss / len(data_loader), infer_accuracy/len(data_loader)

def letterbox(image, **kwargs):
    resized_width, resized_height = kwargs['resized_width'][0], kwargs['resized_height'][0]
    height, width, _ = image.shape
    scale = min(resized_height / height, resized_width / width)
    new_height = int(height * scale)
    new_width = int(width * scale)
    image = cv2.resize(image, (new_width, new_height))
    new_img = np.full((resized_height, resized_width, 3), 128, dtype='uint8')
    # fill new image with the resized image and centered it
    new_img[(resized_height - new_height) // 2:(resized_height - new_height) // 2 + new_height,
            (resized_width - new_width) // 2:(resized_width - new_width) // 2 + new_width,
            :] = image.copy()
    return new_img 

if __name__ == "__main__":
    image_size = 512 
    epochs = 100
    batch_size = 4
    learning_rate = 0.0003
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrain = None # './runs/exp3/seg.pt' #
    
    n_times = 1 # increase the datasize by n times

    save_dirs = './runs/exp-augments-geometry/iteration-1/'
    # albumentation_letterbox = 
    def custom_transform(data, **kwargs):
        kwargs['resized_width'] = image_size,
        kwargs['resized_height'] = image_size,
        return letterbox(image=data, **kwargs)
    
    transformations = A.Compose([
        # A.Resize(image_size, image_size),
        A.Lambda(name='Letter Box', 
            image=custom_transform,
            mask=custom_transform,
            p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.Blur(blur_limit=(1, 5), p=0.6),
        # A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.25),
        # A.RandomBrightnessContrast(p=0.5),
        # A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.6),
        # A.Rotate(limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False, p=1.0),
        # A.SafeRotate(limit=(-45, 45), interpolation=0, border_mode=0, value=(0, 0, 0), p=0.6),
        # A.augmentations.transforms.Normalize
        # A.ChannelShuffle(p=0.8),
        ToTensorV2(),
    ])

    os.makedirs(save_dirs, exist_ok=True)
    log_stream = open(os.path.join(save_dirs, 'train_log.txt'), 'w+')
    log_stream.write(
        f"\n\
        Image Size: {image_size}\n \
        Total Number of Epochs: {epochs}\n\
        Batch Size: {batch_size}\n\
        Learning rate: {learning_rate}\n\
        n_times : {n_times} \n\
        Device : {device}\n\
        pretrain : {pretrain}\n\
        Save directory : {save_dirs}\n"
    )

    log_stream.write('Used augmentation transformations\n')
    for aug_idx, augmentation in enumerate(transformations.to_dict()['transform']['transforms']):
        log_stream.write(f"{aug_idx}. {augmentation['__class_fullname__']}\n")

    # loading the dataset
    football_dataset = FootballDataset('./images', transformations=transformations, n_times=n_times)
    train_valid_test_split_value = list(map(lambda x : int(x * len(football_dataset)), [0.8, 0.15, .05]))
    train_set, valid_set, test_set = torch.utils.data.random_split(football_dataset, lengths=train_valid_test_split_value)
    # defining the dataloader
    train_dataloader, valid_dataloader, test_dataloader = list(map(lambda x : DataLoader(x, batch_size=batch_size), [train_set, valid_set, test_set] ))
    
    # visualize the data 
    from visualization import visualize 
    visualize(train_dataloader.dataset, no_of_images=8)
    
    channel, d_width, d_height = train_dataloader.dataset[0][0].shape  
    image_size = d_width # as d_width = d_height

    # model 
    segmentation_model = SegmentationModel()
    if pretrain != None:
        segmentation_model.load_state_dict(torch.load(pretrain))

    segmentation_model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=learning_rate)
    
    # start training 
    best_loss = np.Inf
    train_metrics= {    
        'loss' : [],
        'accuracy': [],
    }
    valid_metrics = {
        'loss': [],
        'accuracy': [],
    }
    for i in range(epochs):
        loss, accuracy = train(i,
            segmentation_model, batch_size, image_size, train_dataloader,optimizer, device, save_dirs=save_dirs)
        valid_loss, valid_accuracy = valid(i,
            segmentation_model, batch_size, image_size, train_dataloader,optimizer, device, save_dirs=save_dirs)
    
        print(f"Epoch : {i+1} - Loss: {loss}, Acc:{accuracy}%")

        if (i+1)% 2 == 0:
            print('test ipoch')
            train_metrics['loss'].append(loss.item())
            train_metrics['accuracy'].append(accuracy)
            valid_metrics['loss'].append(valid_loss.item())
            valid_metrics['accuracy'].append(valid_accuracy)

        if loss<best_loss:
            os.makedirs(save_dirs, exist_ok=True)
            torch.save(segmentation_model.state_dict(),os.path.join(save_dirs, 'seg.pt'))
            print("[+] Model Updated")
            best_loss=loss
            log_stream.write(f'Best model @ Epoch: {i} | Train loss: {loss} | Valid loss {valid_loss} | Train accuracy: {accuracy} | Valid accuracy: {valid_accuracy}\n')
    print(train_metrics, valid_metrics)

    plt.subplot(1,2,1)
    plt.plot(train_metrics['loss'], label='train')
    plt.plot(valid_metrics['loss'], label='valid')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,2,2)
    plt.plot(train_metrics['accuracy'], label='train')
    plt.plot(valid_metrics['accuracy'], label='valid')
    plt.legend()
    plt.title("Accuracy")

    plt.savefig(os.path.join(save_dirs, f'training_metrics.jpg'))
    log_stream.close()
    ...