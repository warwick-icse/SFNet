import pickle
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class ToPILImage(object):
    def __init__(self, mode=None):
        self.mode = mode

    def __call__(self, image, target=None):
        image = F.to_pil_image(image, self.mode)
        if target is not None:
            target = F.to_pil_image(target, self.mode)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            target = F.to_tensor(target)
        return image, target


class FarmDataset(Dataset):
    def __init__(self, resolution=None, wind_speed=10):
        if resolution is None:
            resolution = [30, 50]
        self.resolution = resolution

        self.transform = Compose([
            ToPILImage(),
            ToTensor()
        ])

        self.flow_field_low = pd.read_csv('generation/U{}_windfarm.csv'.format(str(wind_speed))).values

        # Thses parameters are calculated based on our training and testing dataset
        self.mean_high = 7.628027139607034
        self.mean_low = 7.50061335085812
        self.ss_high = pickle.load(open('./checkpoint/StandardScaler_high_fidelity.pkl', 'rb'))
        self.ss_low = pickle.load(open('./checkpoint/StandardScaler_low_fidelity.pkl', 'rb'))

        # Normalization
        feature = self.flow_field_low.shape[-1]
        self.flow_field_low -= self.mean_low
        self.flow_field_low = self.flow_field_low.reshape((-1, feature))
        self.flow_field_low = self.ss_low.transform(self.flow_field_low)

        self.flow_field_low = self.flow_field_low.reshape(-1, resolution[0], resolution[1], 1).astype(np.float32)

    def __getitem__(self, index):
        low_flow_field = self.flow_field_low[index]

        low_flow_field, high_flow_field = self.transform(low_flow_field)

        return low_flow_field.to(torch.double)

    def __len__(self):
        return self.flow_field_low.shape[0]

