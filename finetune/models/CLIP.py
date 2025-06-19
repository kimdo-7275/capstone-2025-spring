import timm
import numpy as np

import torch
import torch.nn as nn

from .stmem_models import encoder

class ECGCLIP(torch.nn.Module):
    def __init__(self, image_model, signal_model):
        
        super().__init__()
        
        if image_model == 'vit':
            self.image_encoder =  timm.create_model('vit_base_patch16_224', pretrained=True)
            self.image_encoder.head = nn.Identity()
        
        if signal_model == 'stmem':
            self.signal_encoder = encoder.__dict__['st_mem_vit_base'](num_leads=12,
                                                                      seq_len=2250,
                                                                      patch_size=75,
                                                                      num_classes=None)
            pretrain_state_dict = torch.load('st_mem_vit_base_encoder.pth')['model']
            self.signal_encoder.load_state_dict(pretrain_state_dict, strict=True)

        self.proj_image = nn.Linear(768, 512)
        self.proj_signal = nn.Linear(768, 512)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def encode_image(self, image):
        image_emb = self.image_encoder(image)
        proj_image_emb = self.proj_image(image_emb)
        return proj_image_emb
    
    def encode_signal(self, signal):
        signal_emb = self.signal_encoder(signal)
        proj_signal_emb = self.proj_signal(signal_emb)
        return proj_signal_emb
        
    def forward(self, image, signal):
        image_features = self.encode_image(image)
        signal_features = self.encode_signal(signal)
        
        # normalize
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        signal_features = signal_features / signal_features.norm(dim=1, keepdim=True)
        
        # cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ signal_features.t()
        logits_per_signal = logits_per_image.t()
        
        return logits_per_image, logits_per_signal