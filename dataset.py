import os
import glob
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset

"""Dataset classes"""

class HumanArtifact_dataloader(Dataset):
    """
    Human artifact data loader.
    """
    def __init__(self, data_folder, is_train=True):
        self.is_train = is_train
        self._data_folder = data_folder
        self.build_dataset()

    def build_dataset(self):
        if self.is_train:
            self._input_folder = os.path.join(self._data_folder, "train", 'humans')
            self._label_folder = os.path.join(self._data_folder, "train", 'masks')
            self.train_images = sorted(glob.glob(self._input_folder + "/*.png"))
            self.train_labels = sorted(glob.glob(self._label_folder + "/*.png"))
        else:
            self._input_folder = os.path.join(self._data_folder, "test", 'humans')
            self._label_folder = os.path.join(self._data_folder, "test", 'masks')
            self.test_images = sorted(glob.glob(self._input_folder + "/*.png"))
            self.test_labels = sorted(glob.glob(self._label_folder + "/*.png"))
    
    def __len__(self):
        if self.is_train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, idx):
        
        if self.is_train:
            img_path = self.train_images[idx]
            mask_path = self.train_labels[idx]
        else:
            img_path = self.test_images[idx]
            mask_path = self.test_labels[idx]
        
        # Read image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        mask = ImageOps.fit(mask, (256, 512), Image.BICUBIC)
        mask = np.array(mask.convert("RGB"))[:,:,0]
        mask = np.where(mask == 255, 1, 0)
        mask = np.expand_dims(mask, axis=0)
        
        transforms_image = transforms.Compose([transforms.Resize((512, 256)), 
                                               transforms.CenterCrop((512, 256)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        transforms_mask = transforms.Compose([transforms.Resize((512, 256)),
                                              transforms.CenterCrop((512, 256)),
                                              transforms.ToTensor()])
        
        # Convert to torch tensors
        image = transforms_image(image)
        mask = torch.from_numpy(mask)
        
        sample = {'image': image, 
                  'mask': mask}
        
        return sample
