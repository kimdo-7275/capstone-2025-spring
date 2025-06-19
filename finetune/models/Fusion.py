import torch
import torch.nn as nn

from models.CLIP import ECGCLIP

class FusionConcat(nn.Module):
    def __init__(self, clip, num_classes):
        super().__init__()
        self.clip = clip
        self.head = nn.Linear(1024, num_classes)

    def forward(self, image, signal):
        image_features = self.clip.encode_image(image)
        signal_features = self.clip.encode_signal(signal)

        concat = torch.cat((image_features, signal_features), dim=1)
        logit = self.head(concat)

        return logit

class FusionSum(nn.Module):
    def __init__(self, clip, num_classes):
        super().__init__()
        self.image_encoder = clip.image_encoder
        self.signal_encoder = clip.signal_encoder

        self.image_head = nn.Linear(768, num_classes)
        self.signal_head = nn.Linear(768, num_classes)

    def forward(self, image, signal):
        image = self.image_encoder(image)
        signal = self.signal_encoder(signal)

        image = self.image_head(image)
        signal = self.signal_head(signal)

        logit = image + signal

        return logit

def make_fusion_model(self, image_model, signal_model, fusion_method, num_classes, weight_path):
    clip = ECGCLIP(image_model, signal_model)
    clip.load_state_dict(torch.load(weight_path), strict=True)

    if fusion_method == 'concat':
        model = FusionConcat(clip, num_classes)
    elif fusion_method == 'sum':
        model = FusionSum(clip, num_classes)
        
    return model





    