import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.amp import autocast, GradScaler


from utils.dataset import ImageSignal_Dataset
from utils.clip import ECGCLIP

parser = argparse.ArgumentParser()
parser.add_argument('--seed', 
                    type=int, 
                    default=7)
parser.add_argument('--device',
                    default='cuda',
                    type=str)

parser.add_argument('--csv_path',
                    default='your_path',
                    type=str)
parser.add_argument('--image_path',
                    default='your_path',
                    type=str)
parser.add_argument('--signal_path',
                    default='your_path',
                    type=str)

parser.add_argument('--image_model',
                    default='vit',
                    type=str)
parser.add_argument('--signal_model',
                    default='stmem',
                    type=str)

parser.add_argument('--optimizer',
                    default='adamW',
                    type=str)
parser.add_argument('--learning_rate',
                    default=1e-5,
                    type=float)
parser.add_argument('--epochs',
                    default=100,
                    type=int)
parser.add_argument('--batch_size',
                    default=128,
                    type=int)

def main():
    args = parser.parse_args()
    
    # seed
    print(f"Setting seed: {args.seed}")
    seed(args.seed)
    
    # device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # dataset
    csv_file = pd.read_csv(args.csv_path)
    
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    dataset = ImageSignal_Dataset(args.image_path,
                                  args.signal_path, 
                                  csv_file, 
                                  transform, 
                                  original_fs=500)
    
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=12)
    print(f"DataLoader created with batch size: {args.batch_size}")
    
    # model
    model = ECGCLIP(image_model=args.image_model,
                    signal_model=args.signal_model)
    
    # stmem frozen
    # for l, p in model.signal_encoder.named_parameters():
    #     p.requires_grad = False
    
    model.to(device)
    print(f"Model created with image model: {args.image_model} and signal model: {args.signal_model}")
    
    # training
    if args.optimizer == 'adamW':
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        
    loss_image = nn.CrossEntropyLoss()
    loss_signal = nn.CrossEntropyLoss()
    
    scaler = GradScaler()
    
    model.train()
    losses = []
    
    save_path = f"results/lr_{args.learning_rate}_{args.image_model}_{args.signal_model}
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(args.epochs):
        start_epoch = time.time()
        start_iter = time.time()
        total_loss = 0.0

        for idx, (images, signals) in enumerate(dataloader):
            images, signals = images.to(device), signals.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                logits_image, logits_signal = model(images, signals)
                if torch.isnan(logits_image).any() or torch.isnan(logits_signal).any():
                    print("NaN found in logits")

                labels = torch.arange(len(images)).to(device)
                loss = (loss_image(logits_image, labels) + loss_signal(logits_signal, labels)) / 2

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            end_iter = time.time()
            total_iter = end_iter - start_iter
            print(f'\r[{idx+1}/{len(dataloader)}], Loss: {loss.item()}, Time: {int(total_iter//60)}m {int(total_iter%60)}s', end= '')
            
        total_loss /= len(dataloader)
        losses.append(total_loss)
        
        end_epoch = time.time()
        total_epoch = end_epoch - start_epoch
        print(f'\nEpoch [{epoch+1}/{args.epochs}], Loss: {total_loss:.4f}, Time: {int(total_epoch//60)}m {int(total_epoch%60)}s')
        
    try:
        torch.save(model.state_dict(), os.path.join(save_path, f'clip_epoch_{epoch+1}.pth'))
        torch.save(model.image_encoder.state_dict(), os.path.join(save_path, f'{args.image_model}_epoch_{epoch+1}.pth'))
        torch.save(model.signal_encoder.state_dict(), os.path.join(save_path, f'{args.signal_model}_epoch_{epoch+1}.pth'))
        print(f"Model saved at epoch {epoch+1}")
    except Exception as e:
        print(f"Error saving model: {e}")
            
    print("Training completed.")
    
    plt.figure(figsize=(8, 5))
    plt.title('Loss Curve')
    plt.plot(losses, label='train loss')
    plt.xlabel('Epochs')
    
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))
    plt.show()
    
    with open(os.path.join(save_path, "losses.pkl"), "wb") as f:
        pickle.dump(losses, f)

def seed(seed_num):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed_num)
    
    np.random.seed(seed_num)
    
    torch.manual_seed(seed_num)
    torch.cuda.manual_seed_all(seed_num)
    torch.backends.cudnn.benchmark = False
    
if __name__ == '__main__' : main()