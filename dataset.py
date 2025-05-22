import os
import random

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class MNISTHAM10000(Dataset):
    # def __init__(self, root, df, multiply, multiply_factor, augment_classes, transform=None, mask_transform=False):
    def __init__(self, root, df, multiply, multiply_factor, transform=None, mask_transform=False, with_mask = False):
        self.root = root
        self.data = df
        self.multiply = multiply
        # self.multiply_factor = multiply_factor
        # self.augment_classes = augment_classes
        if self.multiply:
            self.multiply_factor = multiply_factor
        
        self.transform = transform
        self.mask_transform = mask_transform
        self.with_mask = with_mask

        # Размножаем строки для определённых классов
        self.augmented_df = df.copy()
        if self.multiply:
            for label in self.multiply_factor[1]:
                class_data = df[df['dx'] == label]
                class_data = pd.concat([class_data] * self.multiply_factor[0][self.multiply_factor[1].index(label)], ignore_index=True)
                self.augmented_df = pd.concat([self.augmented_df, class_data], ignore_index=True)
        
        self.imgs = self.augmented_df['image_id'].values
        if self.with_mask:
            self.masks = self.augmented_df['image_id'].values + '_segmentation'
        self.labels = self.augmented_df['cell_type_idx'].values

    def __getitem__(self, idx):
        if self.with_mask:
        # load images ad masks
            img_path = os.path.join(self.root + 'imgs',self.imgs[idx]+'.jpg')
            img = Image.open(img_path).convert("RGB")
    
            mask_path = os.path.join(self.root + 'masks', self.masks[idx]+'.png')
            mask_img = Image.open(mask_path)
    
            if self.transform:
              img = self.transform(img)
            if self.mask_transform and self.with_mask: 
              mask_img = self.mask_transform(mask_img)
            
            return img, mask_img
            
        else:
            img_path = os.path.join(self.root + 'imgs',self.imgs[idx]+'.jpg')
            img = Image.open(img_path).convert("RGB")
    
            label = torch.tensor(self.labels[idx], dtype=torch.long)
    
            if self.transform:
              img = self.transform(img)
            
            return img, label
        
    def __len__(self):
        return len(self.imgs)


# задаем преобразования для данных
img_transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),  # Convert to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalize
        ])

mask_transform = transforms.Compose([
            transforms.Resize(size=(128, 128)),
            transforms.ToTensor(),  # Convert to PyTorch Tensor
        ])

segment_transform = transforms.Compose([
            transforms.Resize(size=(256, 256)),
            transforms.ToTensor(),  # Convert to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])# Normalize