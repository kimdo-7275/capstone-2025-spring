import os
import time
import timm
import pytz
import argparse
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from utils import seed, evaluate
from datasets import load_ptbxl, load_cpsc, load_csn

from models.Encoder import make_encoder_model
from models.Fusion import make_fusion_model

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name',
                    required= True,
                    type= str,
                    help= '실험 이름 (간략한 설명)')
parser.add_argument('--seed',
                    default= 7,
                    type= int)
parser.add_argument('--gpu',
                    default= 'cuda:0',
                    type= str)

parser.add_argument('--csv_path',
                    default= 'your_path',
                    help= 'train/val/test로 나뉜 csv 파일 경로')
parser.add_argument('--image_path',
                   default= 'your_path',
                   help= 'ECG 이미지 경로 (ecg-img-kit)')
parser.add_argument('--signal_path',
                   default= 'your_path',
                   help= 'ECG 시그널 (physionet.org/files/1.0.3/training/')
parser.add_argument('--weight_path',
                   default= 'your_path',
                   help= '사전 학습된 가중치 파일 경로')
parser.add_argument('--result_dir',
                   default= 'your_path',
                   help= '결과 저장 디렉토리')

parser.add_argument('--data',
                    default= 'ptbxl',
                    type= str)
parser.add_argument('--ptbxl_class',
                    default= 'super_class',
                    type= str)
parser.add_argument('--sampling_rate',
                    default= 500,
                    type= int,
                    help= 'options - ptbxl: 100 or 500')

parser.add_argument('--mode',
                   default= 'fusion',
                   type= str,
                   help= 'fusion / image / signal 중 선택')
parser.add_argument('--fusion_method',
                   default= 'concat',
                   type= str,
                   help= '모달리티 융합 방식 (concat, sum)')
parser.add_argument('--image_model',
                   default= 'vit',
                   type= str,
                   help='이미지 인코더 모델 이름 (vit)')
parser.add_argument('--signal_model',
                   default= 'stmem',
                   type= str,
                   help= '시그널 인코더 모델 이름 (stmem)')

parser.add_argument('--batch_size',
                    default= 16,
                    type= int)
parser.add_argument('--loss_function',
                    default= 'BCEWithLogitLoss',
                    type= str)
parser.add_argument('--optimizer',
                    default= 'AdamW',
                    type= str)
parser.add_argument('--learning_rate',
                    default= 1e-3,
                    type= float)
parser.add_argument('--epochs',
                    default= 2,
                    type= int)


def main():
    # parser #
    args = parser.parse_args()
    
    # exp start #
    print(f'\n[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] start experiment')
    
    # seed #
    seed(args.seed)
    
    # gpu #
    device = torch.device(args.gpu if torch.cuda.is_available() else 'cpu')
    print(f'[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] device: {device}')
    
    # data #
    if args.data == 'ptbxl':
        csv_file = pd.read_csv(os.path.join(args.csv_path, f'{args.ptbxl_class}/ptbxl_{args.ptbxl_class}_test.csv'))
        label_names = list(csv_file.columns[6:])
        num_classes = len(label_names)
        
        trainloader, validloader, testloader = load_ptbxl(args.ptbxl_class,
                                                          args.csv_path,
                                                          args.image_path,
                                                          args.signal_path,
                                                          args.sampling_rate,
                                                          args.batch_size)
    elif args.data == 'cpsc':
        csv_file = pd.read_csv(os.path.join(args.csv_path, 'icbeb_test.csv'))
        label_names = list(csv_file.columns[7:])
        num_classes = len(label_names)

        trainloader, validloader, testloader = load_cpsc(args.csv_path,
                                                         args.image_path,
                                                         args.signal_path,
                                                         args.sampling_rate,
                                                         args.batch_size)
    elif args.data == 'csn':
        csv_file = pd.read_csv(os.path.join(args.csv_path, 'csn_test.csv'))
        label_names = list(csv_file.columns[3:])
        num_classes = len(label_names)

        trainloader, validloader, testloader = load_csn(args.csv_path,
                                                         args.image_path,
                                                         args.signal_path,
                                                         args.sampling_rate,
                                                         args.batch_size)
    
    for images, signals, labels in trainloader:
        print(f'[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] 1st loader, images shape: {images.shape} signals shape: {signals.shape} label shape: {labels.shape}')
        break
    
    # model #
    if args.mode == 'fusion':
        model_name = f'clip_{args.fusion_method}'
        model = make_fusion_model(args.image_model, 
                                  args.signal_model, 
                                  args.fusion_method,
                                  num_classes,
                                  args.weight_path)
    elif args.mode == 'signal':
        model_name = args.signal_model
        model = make_encoder_model(args.signal_model, 
                                   num_classes, 
                                   args.weight_path,
                                   args.gpu)
    elif args.mode == 'image':
        model_name = args.image_model
        model = make_encoder_model(args.image_model, 
                                   num_classes,
                                   args.weight_path,
                                   args.gpu)
    
    print(f'[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] model: {model_name}')
    print(model)

    # lp / ft setting #
    linear_layer_map = {'efficientnet': 'classifier',
                        'convnext': 'head.fc',
                        'resnet50': 'fc',
                        'vit': 'head',
                        'inception': 'head',
                        'stmem': 'head',
                        'densenet': 'classifier'}
    
    if 'lp' in args.exp_name:
        linear_layer_names = linear_layer_map.get(model_name, 'head')
        for l, p in model.named_parameters():
            if linear_layer_names in l:
                print(l)
                p.requires_grad = True
            else:
                p.requires_grad = False
    
    # train setting #
    if args.loss_function == 'BCEWithLogitLoss':
        criterion = nn.BCEWithLogitsLoss()
    
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr= args.learning_rate)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr= args.learning_rate)
    
    ##### result save setting #####
    if args.data == 'ptbxl':
        setting = f'seed_{args.seed}_data_{args.data}_class_{args.ptbxl_class}_sampling_rate_{args.sampling_rate}_batch_size_{args.batch_size}_model_{model_name}_loss_function_{args.loss_function}_optimizer_{args.optimizer}_learning_rate_{args.learning_rate}_epochs_{args.epochs}'
    else:
        setting = f'seed_{args.seed}_data_{args.data}_sampling_rate_{args.sampling_rate}_batch_size_{args.batch_size}_model_{model_name}_loss_function_{args.loss_function}_optimizer_{args.optimizer}_learning_rate_{args.learning_rate}_epochs_{args.epochs}'
        
    result_path = os.path.join(args.result_dir, f'{args.exp_name}/{setting}')

    result_csv_path = os.path.join(result_path, 'results.csv')
    print(result_path)
    os.makedirs(result_path, exist_ok= True)
    
    best_auc = float('-inf')
    best_auc_dict = os.path.join(result_path, 'best_auc_dict.pth')
    
    result_csv_path = f'{args.result_dir}/{args.exp_name}/results.csv'
    if not os.path.exists(result_csv_path):
        new_file = pd.DataFrame(
            columns= ['seed', 'epoch', 'batch', 'model', 'optim', 'lr', 'ACC', 'AUC', 'F1']
        )
        new_file.to_csv(result_csv_path, index= False)
        print(f'[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] make new results csv file')
    ###############################
    
    # train #
    print(f'[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] start train')

    scaler = GradScaler()
    
    train_losses, train_accs, train_aucs, train_f1s = [], [], [], []
    valid_losses, valid_accs, valid_aucs, valid_f1s = [], [], [], []
    
    model.to(device)
    for epoch in range(args.epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        for images, signals, labels in trainloader:
            images, signals, labels = images.to(device), signals.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                if args.mode == 'fusion':
                    outputs = model(images, signals)
                elif args.mode == 'signal':
                    outputs = model(signals)
                elif args.mode == 'image':
                    outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        trainscore = evaluate(device, trainloader, model, args.mode, num_classes)
        train_losses.append(round(train_loss/len(trainloader),4))
        train_accs.append(round(trainscore['acc_avg'], 4))
        train_aucs.append(round(trainscore['auc_avg'], 4))
        train_f1s.append(round(trainscore['f1_avg'], 4))
        
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, signals, labels in validloader:
                images, signals, labels = images.to(device), signals.to(device), labels.to(device)
                
                if args.mode == 'fusion':
                    outputs = model(images, signals)
                elif args.mode == 'signal':
                    outputs = model(signals)
                elif args.mode == 'image':
                    outputs = model(images)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
            validscore = evaluate(device, validloader, model, args.mode, num_classes)
            valid_losses.append(round(valid_loss/len(validloader),4))
            valid_accs.append(round(validscore['acc_avg'], 4))
            valid_aucs.append(round(validscore['auc_avg'], 4))
            valid_f1s.append(round(validscore['f1_avg'], 4))
        
        if best_auc < validscore['auc_avg']:
            best_auc = validscore['auc_avg']
            torch.save(model.state_dict(), best_auc_dict)
        
        end_time = time.time()
        total_time = end_time - start_time
        print(f'[{epoch+1}/{args.epochs}] Time: {int(total_time//60)}m {int(total_time%60)}s, Loss - Train: {train_losses[-1]:.4f} Valid: {valid_losses[-1]:.4f}')
    print(f'\n[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] complete train')
    
    # save result #
    print(f'\n[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] start saving result')
    
    '''train curve img'''
    plt.figure(figsize= (6, 6))
    
    plt.subplot(2,2,1)
    plt.title('Loss curve')
    plt.plot(train_losses, label= 'Train')
    plt.plot(valid_losses, label= 'Valid')
    plt.legend()
    
    plt.subplot(2,2,2)
    plt.title('Acc curve')
    plt.plot(train_accs, label= 'Train')
    plt.plot(valid_accs, label= 'Valid')
    plt.legend()
    
    plt.subplot(2,2,3)
    plt.title('AUC curve')
    plt.plot(train_aucs, label= 'Train')
    plt.plot(valid_aucs, label= 'Valid')
    plt.legend()
    
    plt.subplot(2,2,4)
    plt.title('F1 curve')
    plt.plot(train_f1s, label= 'Train')
    plt.plot(valid_f1s, label= 'Valid')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_path, 'result_curve.png'))
    plt.close()
    
    print(f'[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] complete saving png result')
    
    '''csv result'''
    model.load_state_dict(torch.load(best_auc_dict))
    testscore = evaluate(device, testloader, model, args.mode, num_classes)
    
    new_result = {
        'seed': args.seed,
        'epoch': args.epochs,
        'batch': args.batch_size,
        'optim': args.optimizer,
        'lr': args.learning_rate,
        'model': model_name,
        'ACC': round(testscore['acc_avg'], 4),
        'AUC': round(testscore['auc_avg'], 4),
        'F1': round(testscore['f1_avg'], 4)
    }
    new_result = pd.DataFrame([new_result])

    results = pd.read_csv(result_csv_path)
    if results.empty:
        results = new_result
    else:
        results = pd.concat([results, new_result], ignore_index= True)
    results.to_csv(result_csv_path, index= False, encoding= 'utf-8-sig')
    print(f'[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] complete saving csv result')
    
    # all process done #
    print(f'[{datetime.now(pytz.timezone("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")}] all process complete')
    
if __name__ == '__main__': main()