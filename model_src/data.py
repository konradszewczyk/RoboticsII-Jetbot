import torch
import cv2
import os
import numpy as np
import pandas as pd

from os.path import join
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class JetBotDataset(Dataset):
    def __init__(self, dataset_path: str = 'dataset/augmented', split_type: str = 'train', use_next: bool = True):
        self.dataset_path = dataset_path
        self.use_next = use_next
        self.split_type = split_type

        train_test_df = pd.read_csv(join(dataset_path, 'train_test.csv'), index_col=0)
        split = train_test_df[train_test_df['split'] == split_type]['run_no']

        control_df = pd.read_csv(join(dataset_path, 'control.csv'), index_col=0)
        self.dataset_df = control_df[control_df['run_no'].isin(split)].reset_index(drop=True)

        if split_type == 'test':
            self.dataset_df = self.dataset_df[self.dataset_df['augment_idx'] == 0].reset_index(drop=True)

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        row = self.dataset_df.iloc[idx]
        run_folder = '{:03d}'.format(round(row['run_no']))
        img_name = '{:04d}_{:03d}.jpg'.format(round(row['step_no']), round(row['augment_idx']))
        img_path = join(self.dataset_path, run_folder, img_name)

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = img / 255
        img = torch.from_numpy(img)
        img = transforms.functional.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.use_next:
            control = np.array(row[['forward_next', 'left_next']])
        else:
            control = np.array(row[['forward', 'left']])
        return img, control
