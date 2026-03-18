import pandas as pd
import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MedicalImageDataset(Dataset):
    def __init__(self, metadata_path: str, image_dir: str, transform=None):
        """
        Args:
            metadata_path (str): Path to the excel metadata file.
            image_dir (str): Directory with all the preprocessed images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Note: You will need openpyxl installed to read the excel file (pip install openpyxl).
        self.metadata = pd.read_excel(metadata_path)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # NOTE: You must adjust the column indices depending on the Metadata.xlsx structure.
        # Here we assume the first column (index 0) is the filename/id and second (index 1) is the label.
        img_name = str(self.metadata.iloc[idx, 0])
        
        # Ensure the extension matches the preprocessed images
        if not img_name.endswith('.png'):
            img_name += '.png'
            
        img_path = os.path.join(self.image_dir, img_name)
        
        # Load image using OpenCV and normalize
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
            
        image = image.astype(np.float32) / 255.0
        # Add channel dimension (C, H, W) for PyTorch
        image = np.expand_dims(image, axis=0)

        # Target label (adjust index based on your Excel file)
        label = self.metadata.iloc[idx, 1]

        # Apply optional transforms
        if self.transform:
            # Transforms are typically applied directly if using torchvision,
            # For Albumentations, modify this sequence to: image = self.transform(image=image)["image"]
            image = self.transform(image)
            
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        # Adapt dtype as needed: torch.long for classification, torch.float for regression
        label_tensor = torch.tensor(label, dtype=torch.long)

        return image_tensor, label_tensor

def get_dataloader(metadata_path: str, image_dir: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    """
    Creates and returns a PyTorch DataLoader.
    """
    dataset = MedicalImageDataset(metadata_path, image_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return dataloader

    # Example usage:
    # dl = get_dataloader(
    #     metadata_path="../DMID_PNG/Metadata.xlsx",
    #     image_dir="../DMID_PNG/1024/TIFF_PREPROCESSED/",
    #     batch_size=8
    # )
    # for images, labels in dl:
    #     print(images.shape, labels.shape)
    #     break
