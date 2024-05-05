import os 
import cv2
import numpy as np 
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import albumentations as A 
from albumentations.pytorch import ToTensorV2

import torch  
import torch.utils
from torch import nn 
import torch.utils.data
from torchvision import transforms 
from torch.nn import functional as F 
from torch.utils.data import DataLoader

from model import SegmentationModel
from football_dataset import FootballDataset
from metric import Metrics 

def train(epoch, model: SegmentationModel, batch_size, image_size, data_loader, optimizer,  
          device = 'cuda' if torch.cuda.is_available() else 'cpu', 
          metric: Metrics= None, save_dirs = './runs/exp/'):
    '''
    Function to train the model
    '''
    train_metrics = {
        'loss': 0.0,
        'miou': 0.0,
        'pxl_acc': 0.0,
        'iou_c': {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
        },
    }

    model.train()

    print(len(data_loader))

    # for i, (images, masks) in tqdm(enumerate(data_loader), total= len(data_loader.dataset), bar_format=TQDM_BAR_FORMAT):
    for images, masks in tqdm(data_loader): #, total= len(data_loader.dataset), bar_format=TQDM_BAR_FORMAT):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # pred, loss = model(images, masks)
        pred = model(images)
        metric = metric(pred, masks)
        loss = metric.calculate_loss()
        mIOU, iou_per_class_dict = metric.mIoU()
        pixel_accuracy = metric.pixel_accuracy()

        loss.backward()
        optimizer.step()

        train_metrics['loss'] += loss 
        train_metrics['miou'] += mIOU
        train_metrics['pxl_acc'] += pixel_accuracy
        for key, value in iou_per_class_dict.items():
            train_metrics['iou_c'][key] += value

    if (epoch) %10 ==0 :
        path_to_fig = os.path.join(save_dirs, 'train_pred')
        os.makedirs( path_to_fig, exist_ok=True)

        plt.subplot(1,3,1)
        plt.imshow(images[0].permute(1,2,0).detach().cpu().numpy())
        plt.axis('off')
        plt.title("Image")

        plt.subplot(1,3,2)
        plt.imshow(torch.argmax(F.softmax(pred, dim=1), dim=1).unsqueeze(1)[0].permute(1,2,0).detach().cpu().numpy())
        plt.axis('off')
        plt.title("Prediction")

        plt.subplot(1,3,3)
        plt.imshow(masks[0].permute(1,2,0).detach().cpu().numpy())
        plt.axis('off')
        plt.title("Truth")

        plt.savefig(os.path.join(path_to_fig, f'image_{epoch+1}.jpg'))
        # plt.show()

    train_metrics['loss'] /= len(data_loader)
    train_metrics['miou'] /= len(data_loader)
    train_metrics['pxl_acc'] /= len(data_loader)
    for key, value in train_metrics['iou_c'].items():
            train_metrics['iou_c'][key] /= len(data_loader)

    # print(f"Train loss : {train_metrics['loss']}\nmIOU: {train_metrics['miou']}\n pixel_accuracy {train_metrics['pxl_acc']}")
    return  train_metrics # train_loss / len(data_loader), train_accuracy/len(data_loader)

def valid(epoch, model: SegmentationModel, batch_size, image_size, data_loader, optimizer,  
          device = 'cuda' if torch.cuda.is_available() else 'cpu', 
          metric: Metrics= None, save_dirs = './runs/exp/'):
    '''
    Function for validation
    '''
    valid_metrics = {
        'loss': 0.0,
        'miou': 0.0,
        'pxl_acc': 0.0,
        'iou_c': {
            0: 0.0,
            1: 0.0,
            2: 0.0,
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
            7: 0.0,
            8: 0.0,
            9: 0.0,
        },
    }

    model.eval()
    with torch.inference_mode():
        infer_loss = 0.0
        infer_accuracy = 0.0 

        for images, masks in tqdm(data_loader): #, total= len(data_loader.dataset), bar_format=TQDM_BAR_FORMAT):
            images = images.to(device)
            masks = masks.to(device)

            pred = model(images, masks)
            metric = metric(pred, masks) 
            loss = metric.calculate_loss() 
            mIOU, iou_per_class_dict = metric.mIoU()
            pixel_accuracy = metric.pixel_accuracy() 

            valid_metrics['loss'] += loss 
            valid_metrics['miou'] += mIOU
            valid_metrics['pxl_acc'] += pixel_accuracy
            for key, value in iou_per_class_dict.items():
                valid_metrics['iou_c'][key] += value
            
            # infer_loss += loss 
            # accuracy = ((torch.sum(masks-pred).item())/(batch_size* 3 *image_size*image_size))*100
            # infer_accuracy += (100 - accuracy)
        
        if (epoch + 1 ) %10 ==0 :
            path_to_fig = os.path.join(save_dirs, 'valid_pred')
            os.makedirs( path_to_fig, exist_ok=True)

            plt.subplot(1,3,1)
            plt.imshow(images[0].permute(1,2,0).detach().cpu().numpy())
            plt.axis('off')
            plt.title("Image")

            plt.subplot(1,3,2)
            plt.imshow(torch.argmax(F.softmax(pred, dim=1), dim=1).unsqueeze(1)[0].permute(1,2,0).detach().cpu().numpy())
            plt.axis('off')
            plt.title("Prediction")

            plt.subplot(1,3,3)
            plt.imshow(masks[0].permute(1,2,0).detach().cpu().numpy())
            plt.axis('off')
            plt.title("Truth")

            plt.savefig(os.path.join(path_to_fig, f'image_{epoch+1}.jpg'))
            # plt.show()
        valid_metrics['loss'] /= len(data_loader)
        valid_metrics['miou'] /= len(data_loader)
        valid_metrics['pxl_acc'] /= len(data_loader)

        print(f"Train loss : {valid_metrics['loss']}\nmIOU: {valid_metrics['miou']}\n pixel_accuracy {valid_metrics['pxl_acc']}")
        return valid_metrics # infer_loss / len(data_loader), infer_accuracy/len(data_loader)

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

if __name__ == "__main__":
    color_class_mapping = {
        (137, 126, 126) : 0, # ground
        (27,  71, 151) : 1, # advertisement
        (111,  48, 253): 2, # audience
        (255,   0, 29) : 3, # football post
        (255, 160, 1) : 4, # team A # orange | goal keeper in green of MU
        (255, 159, 0) : 5, #  goal keeper in green of MU keeper
        (254, 233, 3) : 6, # team B # yellow | goal keeper in yellow of RMA
        (255, 235, 0) : 7, #  goal keeper in yellow of RMA
        (238, 171, 171) : 8, # refree pink
        (201,  19, 223) : 9, # football
    }
    class_object_mapping = {
        0 :  "ground",
        1 :  "advertisement",
        2 :  "audience",
        3 :  "football post",
        4 :  "team A", # orange | goal keeper in green of MU
        5 :  "goal keeper A", # green of MU keeper
        6 :  "team B", # yellow | goal keeper in yellow of RMA
        7 :  "goal keeper B", # in yellow of RMA
        8 :  "refree", # pink
        9 :  "football",
    }

    image_size = 512 
    epochs = 50
    batch_size = 4
    learning_rate = 0.0003
    no_of_classes = len(color_class_mapping)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pretrain = None # './runs/exp3/seg.pt' #
    
    n_times = 1 # increase the datasize by n times

    save_dirs = './runs/exp-c-optim/iteration-2'
    # albumentation_letterbox = 
    def custom_transform(data, **kwargs):
        kwargs['resized_width'] = image_size,
        kwargs['resized_height'] = image_size,
        return letterbox(image=data, **kwargs)
    
    transformations = A.Compose([
        A.Resize(image_size, image_size),
        # A.Lambda(name='Letter Box', 
        #     image=custom_transform,
        #     mask=custom_transform,
        #     p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.6),
        A.RandomBrightnessContrast(p=0.5),
        
        A.Blur(blur_limit=(1, 5), p=0.6),
        A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.25),
        # A.Rotate(limit=(-90, 90), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False, p=1.0),
        # A.SafeRotate(limit=(-45, 45), interpolation=0, border_mode=0, value=(0, 0, 0), p=0.6),
        A.ChannelShuffle(p=0.8),
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
    football_dataset = FootballDataset(
        images_path='./images', masks_path='./masks', 
        transformations=transformations, n_times=n_times)
    
    train_valid_test_split_value = list(map(lambda x : int(x * len(football_dataset)), [0.85, 0.14, .01]))
    train_set, valid_set, test_set = torch.utils.data.random_split(football_dataset, lengths=train_valid_test_split_value)
    
    # defining the dataloader
    train_dataloader, valid_dataloader, test_dataloader = list(map(lambda x : DataLoader(x, batch_size=batch_size), [train_set, valid_set, test_set] ))
    
    # visualize the data 
    from visualization import visualize 
    visualize(train_dataloader.dataset, no_of_images=8)
    
    channel, d_width, d_height = train_dataloader.dataset[0][0].shape  
    image_size = d_width # as d_width = d_height

    # model 
    segmentation_model = SegmentationModel(
        encoder='timm-efficientnet-b0', weights='imagenet',
        in_channels=3, classes=no_of_classes)
    
    if pretrain != None:
        segmentation_model.load_state_dict(torch.load(pretrain))
    segmentation_model.to(device)

    # optimizer
    optimizer = torch.optim.Adam(segmentation_model.parameters(), lr=learning_rate)
    # loss 
    loss_func =  nn.CrossEntropyLoss() #(pred, torch.argmax(mask, dim=1))
    # metrics 
    metrics = Metrics(loss_func=loss_func, no_of_class=len(color_class_mapping))

    # start training 
    best_loss = np.Inf
    train_metrics= {    
        'loss': [],
        'miou': [],
        'pxl_acc': [],
    }
    valid_metrics = {
        'loss': [],
        'miou': [],
        'pxl_acc': [],
    }

    for i in range(epochs):
        train_metric = train(i,
            segmentation_model, batch_size, image_size, train_dataloader,optimizer, 
            device, metric=metrics, save_dirs=save_dirs)
        valid_metric = valid(i,
            segmentation_model, batch_size, image_size, valid_dataloader,optimizer, 
            device, metric= metrics, save_dirs=save_dirs)
    
        print(f"Epoch : {i+1}")
        print(f"Train - Loss: {train_metric['loss']}, mIOU:{train_metric['miou']}, pixel accuracy {train_metric['pxl_acc']}")
        print(f"Valid - Loss: {valid_metric['loss']}, mIOU:{valid_metric['miou']}, pixel accuracy {valid_metric['pxl_acc']}")

        if (i+1)% 2 == 0:
            print('test ipoch')
            train_metrics['loss'].append(train_metric['loss'].item())
            train_metrics['miou'].append(train_metric['miou'])
            train_metrics['pxl_acc'].append(train_metric['pxl_acc'])

            valid_metrics['loss'].append(valid_metric['loss'].item())
            valid_metrics['miou'].append(valid_metric['miou'])
            valid_metrics['pxl_acc'].append(valid_metric['pxl_acc'])

        if train_metric['loss'].item() < best_loss:
            os.makedirs(save_dirs, exist_ok=True)
            torch.save(segmentation_model.state_dict(),os.path.join(save_dirs, 'seg.pt'))
            print("[+] Model Updated")
            best_loss=train_metric['loss'].item()
            
            training_logs = f"\n\
            Best model @ Epoch: {i}\n\
            Metrics:\tTrain\tValid\n\
            Loss\t{train_metric['loss'].item():.4f}\t{valid_metric['loss'].item():.4f}\n\
            mIOU\t{train_metric['miou']:.4f}\t{valid_metric['miou']:.4f}\n\
            pxl_acc\t{train_metric['pxl_acc']:.4f}\t{valid_metric['pxl_acc']:.4f}\n\
            "
            
            class_iou_logs = f"\n"
            for key, value in train_metric['iou_c'].items():
                class_iou_logs += f"IOU of {class_object_mapping[key]} Class: {value}\n"
            
            
            print(training_logs)
            print(class_iou_logs)

            log_stream.write(training_logs)
            log_stream.write(class_iou_logs)
    
    print(train_metrics, valid_metrics)

    plt.subplot(1,3,1)
    plt.plot([i for i in range(0, len(train_metrics['loss'])*2, 2)], train_metrics['loss'], label='train')
    plt.plot([i for i in range(0, len(valid_metrics['loss'])*2, 2)], valid_metrics['loss'], label='valid')
    plt.legend()
    plt.title("Loss")

    plt.subplot(1,3,2)
    plt.plot([i for i in range(0, len(train_metrics['miou'])*2, 2)], train_metrics['miou'], label='train')
    plt.plot([i for i in range(0, len(valid_metrics['miou'])*2, 2)], valid_metrics['miou'], label='valid')
    plt.legend()
    plt.title("mIOU")

    plt.subplot(1,3,3)
    plt.plot([i for i in range(0, len(train_metrics['pxl_acc'])*2, 2)], train_metrics['pxl_acc'], label='train')
    plt.plot([i for i in range(0, len(valid_metrics['pxl_acc'])*2, 2)], valid_metrics['pxl_acc'], label='valid')
    plt.legend()
    plt.title("Pixel Accuracy")

    plt.savefig(os.path.join(save_dirs, f'training_metrics.jpg'))
    log_stream.close()
    ...