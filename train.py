import os 
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

    if epoch %10 ==0 :
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

        plt.savefig(os.path.join(path_to_fig, f'image_{epoch}.jpg'))
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
        
        if epoch %10 ==0 :
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

            plt.savefig(os.path.join(path_to_fig, f'image_{epoch}.jpg'))
            # plt.show()
        return infer_loss / len(data_loader), infer_accuracy/len(data_loader)

    






if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 4 
    image_size = 512 
    learning_rate = 0.0003
    epochs = 100
    pretrain =  None # './runs/exp1/seg.pt'#
    save_dirs = './runs/exp2/'


    transformations = A.Compose([
        A.Resize(image_size, image_size),
        # A.augmentations.transforms.Normalize
        ToTensorV2()
    ])

    # loading the dataset
    football_dataset = FootballDataset('./images', transformations=transformations)
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
    for i in range(epochs):
        loss, accuracy = train(i,
            segmentation_model, batch_size, image_size, train_dataloader,optimizer, device, save_dirs=save_dirs)
        loss, accuracy = valid(i,
            segmentation_model, batch_size, image_size, train_dataloader,optimizer, device, save_dirs=save_dirs)
    
        print(f"Epoch : {i+1} - Loss: {loss}, Acc:{accuracy}%")

        if loss<best_loss:
            
            os.makedirs(save_dirs, exist_ok=True)
            torch.save(segmentation_model.state_dict(),os.path.join(save_dirs, 'seg.pt'))
            print("Model Updated")
            best_loss=loss
    ...