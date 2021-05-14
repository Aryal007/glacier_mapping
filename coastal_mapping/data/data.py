#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:24:56 2021

@author: mibook
"""
import glob
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

def fetch_loaders(processed_dir, batch_size=32, use_channels=[0,1,2,3],
                  train_folder='train', val_folder='val', test_folder='',
                  shuffle=True):
    """ Function to fetch dataLoaders for the Training / Validation
    Args:
        processed_dir(str): Directory with the processed data
        batch_size(int): The size of each batch during training. Defaults to 32.
    Return:
        Returns train and val dataloaders
    """
    normalize = False
    train_dataset = CoastalDataset(processed_dir / train_folder, use_channels, normalize)
    val_dataset = CoastalDataset(processed_dir / val_folder, use_channels, normalize)
    
    loader = {
        "train": DataLoader(train_dataset, batch_size=batch_size,
                            num_workers=8, shuffle=shuffle),
        "val": DataLoader(val_dataset, batch_size=batch_size,
                          num_workers=3, shuffle=shuffle)}

    if test_folder:
        test_dataset = CoastalDataset(processed_dir / test_folder, use_channels)
        loader["test"] = DataLoader(test_dataset, batch_size=batch_size,
                                    num_workers=3, shuffle=shuffle)

    return loader

class CoastalDataset(Dataset):
    """Custom Dataset for Coastal Data
    Indexing the i^th element returns the underlying image and the associated
    binary mask
    """

    def __init__(self, folder_path, use_channels, normalize=True):
        """Initialize dataset.
        Args:
            folder_path(str): A path to data directory
        """

        self.img_files = glob.glob(os.path.join(folder_path, '*tiff*'))
        self.mask_files = [s.replace("tiff", "mask") for s in self.img_files]
        self.use_channels = use_channels
        self.normalize = normalize
        if self.normalize:
            arr = np.load(folder_path.parent / "normalize.npy")
            self.mean, self.std = arr[0][use_channels], arr[1][use_channels]

    def __getitem__(self, index):

        """ getitem method to retrieve a single instance of the dataset
        Args:
            index(int): Index identifier of the data instance
        Return:
            data(x) and corresponding label(y)
        """

        img_path = self.img_files[index]
        mask_path = self.mask_files[index]
        data = np.load(img_path)
        data = data[:,:,self.use_channels]
        if self.normalize:
            data = (data - self.mean) / self.std
        label = np.load(mask_path)
        
        return torch.from_numpy(data).float(), torch.from_numpy(label).float()

    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)
        """
        return len(self.img_files)