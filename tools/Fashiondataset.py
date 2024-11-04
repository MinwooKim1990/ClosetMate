from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import pickle

class FashionDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None, seed=2024):
        self.dataframe = dataframe.sort_values('Name').reset_index(drop=True)  
        self.image_folder = image_folder
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # 224 size for ViT model
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomRotation(15),  
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.indices = np.arange(len(self.dataframe))
        self.seed = seed

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Name']
        img_path = os.path.join(self.image_folder, f"{img_name}.jpg") 

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)  
        
        return image
    
    def save_dataset(self, user_name):
        save_data = {
            'dataframe': self.dataframe,
            'image_folder': self.image_folder,
            'indices': self.indices,
            'seed': self.seed
        }
        
        save_path = f'saved/{user_name}_dataset.pkl'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        print(f"Dataset saved to {save_path}")
    
    @classmethod
    def load_dataset(cls, user_name):
        load_path = f'saved/{user_name}_dataset.pkl'
        
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        dataset = cls(
            dataframe=save_data['dataframe'],
            image_folder=save_data['image_folder'],
            seed=save_data['seed']
        )
        dataset.indices = save_data['indices']
        
        return dataset

    def show_images(self, indices):
        # Ensure indices is a list
        if not isinstance(indices, list):
            raise TypeError("indices should be a list of integers.")
        
        # Determine grid dimensions
        num_images = len(indices)
        cols = 5
        rows = math.ceil(num_images / cols)
        
        # Create the subplot grid
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = axes.flatten()  # Flatten for easy indexing
        i=1
        for ax, idx in zip(axes, indices):
            try:
                # Retrieve the image path and open the image
                img_name = self.dataframe.iloc[idx]['Name']
                img_path = os.path.join(self.image_folder, f"{img_name}.jpg")
                image = Image.open(img_path).convert('RGB')
                
                # Display the image
                ax.imshow(image)
                ax.axis('off')
                ax.set_title(f"Top {i}")
                i+=1
            except Exception as e:
                # Handle any errors in loading or displaying the image
                ax.axis('off')
                ax.set_title("Image not found")

        # Turn off any remaining empty subplot axes
        for ax in axes[len(indices):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()