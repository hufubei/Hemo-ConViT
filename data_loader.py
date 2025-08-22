# src/data_loader.py

import os
import torch
import pandas as pd
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, Subset
from PIL import Image


def crop_to_roi(img_tensor: torch.Tensor, padding=5) -> torch.Tensor:
    """Crops the image tensor to the non-black region of interest (ROI)."""
    # Find all non-black pixels
    non_black_mask = torch.sum(img_tensor, dim=0) > 0.01
    if not torch.any(non_black_mask):
        return img_tensor # Return original image if it's all black

    # Get the bounding box of the non-black region
    non_black_coords = torch.argwhere(non_black_mask)
    min_h, min_w = torch.min(non_black_coords, dim=0).values
    max_h, max_w = torch.max(non_black_coords, dim=0).values

    # Apply padding safely
    min_h = max(0, min_h - padding)
    min_w = max(0, min_w - padding)
    max_h = min(img_tensor.shape[1] - 1, max_h + padding)
    max_w = min(img_tensor.shape[2] - 1, max_w + padding)

    cropped_img = img_tensor[:, min_h:max_h+1, min_w:max_w+1]
    return cropped_img


class AnemiaDataset(Dataset):
    """Custom Dataset for loading eye images and Hgb values from an Excel file."""
    def __init__(self, excel_file, img_dir, image_size, hgb_anemia_threshold):
        self.img_dir = img_dir
        self.image_size = image_size
        self.hgb_anemia_threshold = hgb_anemia_threshold

        print("Loading and processing dataset (with ROI cropping and resizing)...")
        # Define initial transforms (run on CPU during data loading)
        self.preprocess_transform = transforms.Compose([
            transforms.ToTensor(),
            crop_to_roi,
            transforms.Resize((self.image_size, self.image_size), antialias=True)
        ])

        # Load and clean the metadata from Excel
        df_raw = pd.read_excel(excel_file, dtype=str)
        df_raw['Hgb'] = pd.to_numeric(df_raw['Hgb'], errors='coerce')

        # Filter for valid rows where the image file exists and Hgb is not null
        valid_rows = []
        for _, row in df_raw.iterrows():
            img_number_str = str(row.get('Number', '')).strip().split('.')[0]
            img_path = os.path.join(self.img_dir, f"{img_number_str}.jpg")
            if pd.notna(row['Hgb']) and os.path.exists(img_path):
                valid_rows.append(row.to_dict())
        
        self.data = pd.DataFrame(valid_rows)
        if self.data.empty:
            raise ValueError("ERROR! No valid image/Hgb records were loaded. Check paths and Excel file.")
            
        self.data['is_anemic'] = (self.data['Hgb'] < self.hgb_anemia_threshold).astype(int)
        print(f"Dataset loaded: {len(self.data)} valid samples found.")
        print(f"Hgb range: {self.data['Hgb'].min():.1f} - {self.data['Hgb'].max():.1f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_num_str = str(row["Number"]).strip().split('.')[0]
        hgb_val = float(row["Hgb"])
        
        img_path = os.path.join(self.img_dir, f"{img_num_str}.jpg")
        
        try:
            image_pil = Image.open(img_path).convert("RGB")
            img_tensor = self.preprocess_transform(image_pil)
        except Exception as e:
            print(f"Warning: Could not load/process image {img_path}. Error: {e}. Returning a black tensor.")
            img_tensor = torch.zeros((3, self.image_size, self.image_size))
            
        return img_tensor, torch.tensor(hgb_val, dtype=torch.float32)


class ListDataset(Dataset):
    """A simple dataset wrapper for a list of (image, label) tuples."""
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]


def create_augmented_and_balanced_dataset(original_train_dataset: Subset, threshold: float) -> ListDataset:
    """Augments training data with horizontal flips and balances classes via undersampling."""
    print("\n--- Augmenting and Balancing Training Set ---")
    
    # Augment by adding a horizontally flipped version of each image
    augmented_data = []
    for img_tensor, hgb_tensor in original_train_dataset:
        augmented_data.append((img_tensor, hgb_tensor))
        augmented_data.append((TF.hflip(img_tensor), hgb_tensor))

    # Separate samples into anemic and non-anemic based on the threshold
    anemic_samples = [s for s in augmented_data if s[1].item() < threshold]
    non_anemic_samples = [s for s in augmented_data if s[1].item() >= threshold]

    print(f"\nClass distribution in AUGMENTED training set (before balancing):")
    print(f"  - Anemic samples (< {threshold} Hgb):    {len(anemic_samples)}")
    print(f"  - Non-Anemic samples (>= {threshold} Hgb): {len(non_anemic_samples)}")

    # Undersample the majority class to match the size of the minority class
    n_minority = min(len(anemic_samples), len(non_anemic_samples))
    if len(anemic_samples) > n_minority:
        anemic_samples = random.sample(anemic_samples, n_minority)
    elif len(non_anemic_samples) > n_minority:
        non_anemic_samples = random.sample(non_anemic_samples, n_minority)

    # Combine and shuffle the balanced dataset
    balanced_data = anemic_samples + non_anemic_samples
    random.shuffle(balanced_data)
    
    print(f"\nFinal balanced training set size: {len(balanced_data)}")
    return ListDataset(balanced_data)