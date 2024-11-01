import torch
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
import numpy as np

def extract_features(dataloader, feature_extractor, device):
    feature_extractor.eval()
    features_list = []
    
    with torch.no_grad():
        for images in tqdm(dataloader, desc="특징 추출"):
            images = images.to(device)
            features = feature_extractor(images)
            features_list.append(features.cpu())
            
    features_array = torch.cat(features_list).numpy()
    return features_array

def extract_features_from_images(image_paths, feature_extractor, device, transform=True):
    feature_extractor.eval()
    features_list = []
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    with torch.no_grad():
        for img_path in tqdm(image_paths, desc="Feature extraction"):
            image = Image.open(img_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            features = feature_extractor(image)
            features = features.view(features.size(0), -1)

            features = features.cpu().numpy()
            features_list.append(features)
    
    features_array = np.concatenate(features_list, axis=0)
    return features_array