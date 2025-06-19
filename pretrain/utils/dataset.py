import os
import wfdb
import pandas as pd

import torch
from torch.utils.data import Dataset

from PIL import Image
from scipy.signal import resample

"""
parameters:
    - image_path: ecg-image-kit 이미지가 저장되어 있는 경로 (상위)
    - signal_path: 원본 신호 데이터가 저장되어 있는 경로 (상위)
    - csv_file: 파일 경로가 담긴 csv 파일 (하위)
    - transform: 이미지 전처리
    - original_fs: 원본 신호의 샘플링 주파수
    - target_fs: 리샘플링을 위한 샘플링 주파수 (default: 250)
"""

class ImageSignal_Dataset(Dataset):
    def __init__(self, image_path, signal_path, csv_file, transform, original_fs, target_fs=250):

        self.image_path = image_path
        self.signal_path = signal_path
        
        self.file_path = list(csv_file['path'])
        
        self.transform = transform
        self.original_fs = original_fs
        self.target_fs = target_fs
        
        self.crop_area = (0, 450, 2200, 1700-70)

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        
        file = self.file_path[idx][6:]
        image_path = os.path.join(self.image_path, f'{file}-0.png')
        signal_path = os.path.join(self.signal_path, file)
        
        # image
        image = Image.open(image_path).convert('RGB')
        image = image.crop(self.crop_area)
        image = self.transform(image)
        
        # signal
        signal = wfdb.rdsamp(signal_path)[0]
        signal = signal.T
        signal = resample(signal, int(5000 * self.target_fs / self.original_fs), axis= 1)
        signal = signal[:,:2250]
        
        return image, torch.tensor(signal, dtype= torch.float)