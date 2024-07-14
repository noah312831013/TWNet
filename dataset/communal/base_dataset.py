"""
@Date: 2021/07/26
@description:
"""
import numpy as np
import torch
from dataset.communal.data_augmentation import PanoDataAugmentation

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, mode, shape=None, aug=None):
        if shape is None:
            shape = [512, 1024]

        assert mode == 'train' or mode == 'val' or mode == 'test' or mode is None, 'unknown mode!'
        self.mode = mode
        self.shape = shape
        self.pano_aug = None if aug is None or mode == 'val' else PanoDataAugmentation(aug)
        self.data = None

    def __len__(self):
        return len(self.data)

    def process_data(self, label, image):

        corners = label['vp_aligned_corners']
        TWV = label['TWV']

        if self.pano_aug is not None:
            corners, image,TWV = self.pano_aug.execute_aug(corners, image, TWV)
        eps = 1e-3
        corners[:, 1] = np.clip(corners[:, 1], 0.5+eps, 1-eps)

        output = {}

        output['image'] = image.transpose(2, 0, 1)
        output['TWV']=TWV
        return output
