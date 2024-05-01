import os 
import cv2 
import glob 

import torch 
from torch.utils.data import random_split, Dataset, DataLoader

class FootballDataset(Dataset):

    def __init__(self, path_to_data, transformations = None) -> None:
        super().__init__()

        self.image_paths = sorted(glob.glob(os.path.join(path_to_data, '*.jpg')))
        self.mask_paths = sorted(glob.glob(os.path.join(path_to_data, '*_fuse.png')))
        self.transformations = transformations
        self.number_of_classes = 11 

        assert len(self.image_paths) == len(self.mask_paths), f"Error image lengthe -- {len(self.image_paths)} and mask length -- {len(self.mask_paths)} mismatch"

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])[:,:,::-1] # importing rgb
        mask = cv2.imread(self.mask_paths[index])[:,:,::-1] # importing rgb
        if self.transformations:
            image, mask = self.apply_transformations(image, mask)
        return image/ 255.0, mask.permute(2, 0, 1) / 255.0

    def apply_transformations(self, image, mask):
        transformed = self.transformations(image = image, mask = mask)
        return transformed['image'], transformed['mask']
    
if __name__ == "__main__":
    football_dataset = FootballDataset('./images')
    train_valid_test_split_value = list(map(lambda x : int(x * len(football_dataset)), [0.8, 0.15, .05]))
    train_set, valid_set, test_set = random_split(football_dataset, lengths=train_valid_test_split_value)
    batch_size = 4 
    # defining the dataloader
    train_dataloader, valid_dataloader, test_dataloader = list(map(lambda x : DataLoader(x, batch_size=batch_size), [train_set, valid_set, test_set] ))
    from visualization import visualize 
    visualize(train_dataloader.dataset, no_of_images=8)