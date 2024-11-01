import timm
import torch
import torch.hub 
import torch.nn as nn
import torchvision

class FeatureExtractor(nn.Module):
    def __init__(self, model_type='resnet'):
        super().__init__()
        self.model_type = model_type
        
        if model_type == 'resnet':
            self.model = torchvision.models.resnet101(pretrained=True)
            self.model = nn.Sequential(*(list(self.model.children())[:-1]))
            self.feature_dim = 2048
            
        elif model_type == 'vit':
            self.model = timm.create_model('vit_large_patch16_224', pretrained=True)
            self.model.head = nn.Identity()
            self.feature_dim = 1024
            
        elif model_type == 'deit':
            self.model = timm.create_model('deit_base_patch16_224', pretrained=True)
            self.model.head = nn.Identity()
            self.feature_dim = 768

        elif model_type == 'dino':
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.model.head = nn.Identity()
            self.feature_dim = 768 
            
        else:
            raise ValueError(f"지원되지 않는 모델 타입입니다: {model_type}")
    
    def forward(self, x):
        features = self.model(x)
        if self.model_type == 'resnet':
            features = features.view(features.size(0), -1)
        return features