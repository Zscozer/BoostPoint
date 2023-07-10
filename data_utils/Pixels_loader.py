import glob
import os
import random

import cv2
import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms


def load_pixels_data():
    BASE_DIR = '/home/zhg'
    DATA_DIR = os.path.join(BASE_DIR, 'dataset')
    all_filepath = []

    for cls in glob.glob(os.path.join(DATA_DIR, 'ShapeNet55_superpixels/*')):
        pcs = glob.glob(os.path.join(cls, '*'))
        # pcs = glob.glob(os.path.join(cls))
        all_filepath += pcs
    # print("all_filepath",all_filepath)
    return all_filepath



class PixelsLoader(Dataset):
    def __init__(self,root='/home/zhg/dataset/ShapeNet55_superpixels'):
        self.root = root
        self.all_path = load_pixels_data()
        self.transform1 = transforms.Compose([
            transforms.ToTensor(),
        ])
    def __getitem__(self, item):
        # print("ok")
        pcd_path = self.all_path[item]

        data = np.load(pcd_path,allow_pickle=True)
        data = data.item()
        pixels = data["pixels"]
        # print("pixels",pixels.shape)
        label = data["label"]

        pixels = self.transform1(pixels).squeeze()
        # print("pixels",pixels.shape)
        return pixels, label

    def __len__(self):
        return len(self.all_path)
