from torch.utils.data import Dataset
import torch
import sys
import os
import cv2
import numpy as np


class WFLW(Dataset):
    """
        return : torch.tensor(img, dtype=torch.float32), torch.tensor(heatmap_label, dtype=torch.float32, )
    """

    def __init__(self, mode='train', path=None):
        super().__init__()
        train_path = '/home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/list_98pt_rect_attr_train_test/resplit_0.1_train.txt'
        test_path = '/home/ubuntu/FaceDatasets/WFLW/WFLW_crop_annotations/list_98pt_rect_attr_train_test/resplit_0.1_test.txt'
        path = path if path else train_path if mode is 'train' else test_path
        # Load landmarks as numpy ()
        self.landmarks = np.genfromtxt(path)
        # Load img and its boundaries heatmap
        self.imgs = []
        self.heatmaps = []
        with open(path, 'r') as file:
            for _ in range(self.landmarks.shape[0]):
                info = file.readline().split(' ')
                self.imgs.append(np.load(info[-2]))
                self.heatmaps.append(np.load(info[-1][:-1]))


    def __len__(self):
        return self.landmarks.shape[0]

    def __getitem__(self, index):
        img, heatmap = self.imgs[index], self.heatmaps[index]
        landmarks = self.landmarks[index, :98 * 2]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(heatmap, dtype=torch.float32), torch.FloatTensor(
            list(landmarks))
