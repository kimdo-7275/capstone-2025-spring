import os
import wfdb
import pandas as pd

from PIL import Image
from scipy.signal import resample

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class ImageSignalDataset(Dataset):
    def __init__(self, ptbxl_class, csv_path, split, image_path, signal_path, transform, original_fs, target_fs=250):

        csv_path = os.path.join(csv_path, ptbxl_class)
        csv_file = pd.read_csv(os.path.join(csv_path, f'ptbxl_{ptbxl_class}_{split}.csv'))
        
        self.image_path = image_path
        self.signal_path = signal_path

        self.label_name = list(csv_file.columns[6:])
        self.num_classes = len(self.label_name)
        self.labels = csv_file.iloc[:, 6:].values
        
        self.file_path = list(csv_file['filename_hr'])
        
        self.transform = transform
        self.original_fs = original_fs
        self.target_fs = target_fs
        
        self.crop_area = (0, 450, 2200, 1700-70)

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, idx):
        
        file = self.file_path[idx].split('/')
        image_path = os.path.join(self.image_path, f'{file[1]}/{file[2]}-0.png')
        signal_path = os.path.join(self.signal_path, f'{file[0]}/{file[1]}/{file[2]}')
        
        # image
        image = Image.open(image_path).convert('RGB')
        image = image.crop(self.crop_area)
        image = self.transform(image)
        
        # signal
        signal = wfdb.rdsamp(signal_path)[0]
        signal = signal.T
        signal = resample(signal, int(5000 * self.target_fs / self.original_fs), axis= 1)
        signal = signal[:,:2250]
        signal = torch.tensor(signal, dtype= torch.float)

        # target
        target = self.labels[idx]
        target = torch.tensor(target, dtype= torch.float)
        
        return image, signal, target
        
def load_ptbxl(ptbxl_class, csv_path, image_path, signal_path, fs, batch_size):

    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    trainloader = DataLoader(ImageSignalDataset(ptbxl_class, csv_path, "train", image_path, signal_path, transform, original_fs=fs),
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=12)
    validloader = DataLoader(ImageSignalDataset(ptbxl_class, csv_path, "val", image_path, signal_path, transform, original_fs=fs),
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=12)
    testloader = DataLoader(ImageSignalDataset(ptbxl_class, csv_path, "test", image_path, signal_path, transform, original_fs=fs),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=12)

    return trainloader, validloader, testloader



    