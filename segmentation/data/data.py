#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 13:24:56 2021

@author: mibook
"""
import glob
import os, pdb, gc
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import elasticdeform
from torchvision import transforms

def fetch_loaders(processed_dir, batch_size=32, use_channels=[0,1],
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
    train_dataset = CoastalDataset(processed_dir / train_folder, use_channels, normalize,
                                    transforms = transforms.Compose([
                                               DropoutChannels(0.2),
                                               FlipHorizontal(0.2),
                                               FlipVertical(0.2),
                                               Rot270(0.2),
                                               Cut(0.2)
                                           ])
                                    )
    val_dataset = CoastalDataset(processed_dir / val_folder, use_channels, normalize)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                        num_workers=0, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                        num_workers=0, shuffle=shuffle)
    del(train_dataset)
    del(val_dataset)
    gc.collect()
    return train_loader, val_loader

class CoastalDataset(Dataset):
    """Custom Dataset for Coastal Data
    Indexing the i^th element returns the underlying image and the associated
    binary mask
    """

    def __init__(self, folder_path, use_channels, normalize=False, transforms=None):
        """Initialize dataset.
        Args:
            folder_path(str): A path to data directory
        """

        self.img_files = glob.glob(os.path.join(folder_path, '*tiff*'))
        self.mask_files = [s.replace("tiff", "mask") for s in self.img_files]
        self.use_channels = use_channels
        self.normalize = normalize
        self.transforms = transforms
        if self.normalize:
            arr = np.load(folder_path.parent / "normalize.npy")
            self.mean, self.std = arr[0][use_channels], arr[1][use_channels]
        self.min = np.asarray([0,0,0,0,0,0,0,0,0,0,0,-1,-1])
        self.max = np.asarray([255,255,255,255,255,255, 255, 255,
                                255, 90, 1, 1, 1])

    def __getitem__(self, index):

        """ getitem method to retrieve a single instance of the dataset
        Args:
            index(int): Index identifier of the data instance
        Return:
            data(x) and corresponding label(y)
        """
        data = np.load(self.img_files[index])
        data = data[:,:,self.use_channels]  
        data = (data - self.min) / (self.max - self.min)
        label = np.expand_dims(np.load(self.mask_files[index]), axis=2)
        ones = label == 1
        twos = label == 2
        zeros = np.invert(ones+twos)
        label = np.concatenate((zeros, ones, twos), axis=2)
        del(ones)
        del(twos)
        del(zeros)
        label[np.sum(data[:, :, :7], axis=2) == 0] = 0
        if self.transforms:
            sample = {'image': data, 'mask': label}
            del(data)
            del(label)
            sample = self.transforms(sample)
            data = torch.from_numpy(sample['image'].copy()).float()
            label = torch.from_numpy(sample['mask'].copy()).float()
            del(sample)
        else:
            data = torch.from_numpy(data).float()
            label = torch.from_numpy(label).float()
        gc.collect()
        return data, label

    def __len__(self):
        """ Function to return the length of the dataset
            Args:
                None
            Return:
                len(img_files)(int): The length of the dataset (img_files)
        """
        return len(self.img_files)

class FlipHorizontal(object):
    """Flip horizontal randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipHorizontal
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data[:, ::-1, :]
            label = label[:, ::-1, :]
        return {'image': data, 'mask': label}

class FlipVertical(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipVertical
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data[::-1, :, :]
            label = label[::-1, :, :]
        return {'image': data, 'mask': label}

class Rot270(object):
    """Flip vertically randomly the image in a sample.

    Args:
        p (float between 0 and 1): Probability of Rot270
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            data = data.transpose((1,0,2))
            label = label.transpose((1,0,2))
        return {'image': data, 'mask': label}

class Cut(object):
    """Cut randomly the first 7 channels of image in a sample.

    Args:
        p (float between 0 and 1): Probability of FlipHorizontal
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        channels = data.shape[2]
        if torch.rand(1) < self.p:
            prob = torch.rand(1)
            if prob <= 0.25:
                data[:256, :256, :channels] = 0
                label[:256, :256, :] = 0
            elif prob <= 0.5:
                data[:256, 256:, :channels] = 0
                label[:256, 256:, :] = 0
            elif prob <= 0.75:
                data[256:, :256, :channels] = 0
                label[256:, :256, :] = 0
            else:
                data[256:, 256:, :channels] = 0
                label[256:, 256:, :] = 0
        return {'image': data, 'mask': label}

class ElasticDeform(object):
    """Apply Elasticdeform from U-Net
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            [data, label] = elasticdeform.deform_random_grid([data, label], axis=(0, 1))
        return {'image': data, 'mask': label}

class DropoutChannels(object):
    """Apply Random channel dropouts
    """
    def __init__(self, p):
        if (p > 1) or (p < 0):
            raise Exception("Probability should be between 0 and 1")
        self.p = p

    def __call__(self, sample):
        data, label = sample['image'], sample['mask']
        if torch.rand(1) < self.p:
            rand_channel_index = np.random.randint(low = 0, high = data.shape[2])
            data[:, :, rand_channel_index] = 0
        return {'image': data, 'mask': label}
