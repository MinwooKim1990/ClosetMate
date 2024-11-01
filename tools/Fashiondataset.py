from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt

class FashionDataset(Dataset):
    def __init__(self, dataframe, image_folder, transform=None):
        self.dataframe = dataframe  
        self.image_folder = image_folder
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),  # 224 size for ViT model
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),  
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Name']
        img_path = os.path.join(self.image_folder, f"{img_name}.jpg") 

        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)  
        
        return image

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