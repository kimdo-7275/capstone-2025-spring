import timm

import torch
import torch.nn as nn
from torchvision import models

from models.CNN import Simple_1DCNN
from models.stmem_models import encoder

def make_encoder_model(model_name, num_classes, weight_path, device):
    if model_name == 'cnn':
        model = Simple_1DCNN()
    elif model_name == 'stmem':
        model = encoder.__dict__['st_mem_vit_base'](num_leads=12, 
                                                    seq_len=2250, 
                                                    patch_size=75,
                                                    num_classes=num_classes)
        model.load_state_dict(torch.load('/home/ksh114612/project/finetune/pretrain_weight/st_mem_vit_base_encoder.pth')['model'], strict=False)
        
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', # 86,567,656 params
                                  pretrained=True,
                                  num_classes=num_classes)
        
    elif model_name == 'efficientnet':
        model = timm.create_model('efficientnet_b5', # 30,389,784 params
                          pretrained=True,
                          num_classes=num_classes)
        
    elif model_name == 'convnext':
        model = timm.create_model('convnext_base', # 88,591,464 params
                          pretrained=True,
                          num_classes=num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == 'inception':
        model = timm.create_model('inception_next_base', # 86,672,136 params
                          pretrained=True,
                          num_classes=num_classes)
        
    elif model_name == 'densenet':
        model = timm.create_model('densenet121', # 7,978,856 params
                                pretrained=True,
                                num_classes=num_classes)

    elif model_name == 'resnet':
        model = timm.create_model('resnet50',
                                 pretrained=True, 
                                 num_claseses=num_classes)
        
    if weight_path != 'None':
        model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

    return model