#%%
import os
import cv2
import numpy as np   
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import albumentations as A
from sklearn.model_selection import train_test_split
import pandas as pd
import torchvision.transforms as T
from PIL import Image
import torch.nn as nn
from albumentations.pytorch.transforms import ToTensorV2

os.listdir("../bdappv/ign/mask")


# %%

all_images = os.listdir("../bdappv/ign/img")
print(len(all_images))
masks = os.listdir("../bdappv/ign/mask")

positives = [name for name in all_images if name in masks]

print(masks[1])
print(positives[1])

""" 
Need to split all_images and masks into train/val/test
"""

classification_data = pd.DataFrame(masks)

#%%
X_train, X_test = train_test_split(classification_data, test_size=0.33, random_state=42)

train_images = X_train[0].values.tolist()
test_images = X_test[0].values.tolist()

print(train_images)
print(test_images)

# %%
"""
Insert transforms here
"""

train_transforms = A.Compose([
                A.Flip(p=0.5),
                A.RandomBrightnessContrast(p=0.1),
                #A.augmentations.transforms.ColorJitter(),
                A.Normalize(
                  mean=[0, 0, 0],
                  std=[1.0, 1.0, 1.0],
                ),
                ToTensorV2(),
            ])

test_transforms = A.Compose([
                A.Normalize(
                  mean=[0, 0, 0],
                  std=[1.0, 1.0, 1.0],
                ),
                ToTensorV2(),
            ])

# %%
class SolarSegmentationDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = f"../bdappv/ign/img/{img_name}"
        mask_path = f"../bdappv/ign/mask/{img_name}"
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path)) / 255.0
        transformed = self.transforms(image=image, mask=mask)
        image_out = transformed["image"]
        mask_out = transformed["mask"].unsqueeze(0)
        print(image_out.shape)
        print(mask_out.shape)
        return image_out, mask_out


train_dataset = SolarSegmentationDataset(train_images, train_transforms)
test_dataset = SolarSegmentationDataset(test_images, test_transforms)

print(len(train_dataset))
print(len(test_dataset))

image, label = train_dataset.__getitem__(1)

# %%
print(torch.unique(label))

# %%

from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

num_classes = 1
device = torch.device("mps")
model = deeplabv3_resnet50(pretrained = True, progress = True)
model.classifier = DeepLabHead(2048, num_classes)
model.to(device)

test = image.unsqueeze(0).to(device)

print(test.shape)

model.eval()
out = model(test)

print(out["out"].shape)


#%%
criterion = nn.BCEWithLogitsLoss()

label = label.float().to(device).unsqueeze(0)

loss = criterion(out["out"], label)
# %%

print(loss.item())

# %%
