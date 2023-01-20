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
os.listdir("../bdappv/ign/mask")

out = cv2.imread("../bdappv/ign/mask/POGJA58D5VDIWV.png")

#%%
import torch
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print (x)
else:
    print ("MPS device not found.")

# %%
all_images = os.listdir("../bdappv/ign/img")
print(len(all_images))
masks = os.listdir("../bdappv/ign/mask")

negatives = [name for name in all_images if name not in masks]
positives = [name for name in all_images if name not in negatives]
print(len(positives))
print(len(negatives))
print(len(all_images))

classification_data = {}

for name in all_images:
    if name in positives:
        classification_data[name] = 1
    elif name in negatives:
        classification_data[name] = 0
""" 
Need to split all_images and masks into train/val/test
"""

classification_data = pd.DataFrame(classification_data.items())

#%%
print(classification_data[0])
#%%

X_train, X_test, y_train, y_test = train_test_split(classification_data[0], classification_data[1], test_size=0.33, random_state=42)

train_images = X_train.tolist()
train_labels = y_train.tolist()
test_images = X_test.tolist()
test_labels = y_test.tolist()

#%%

pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

train_transforms = T.Compose([
                        T.ToTensor(),
                        T.RandomHorizontalFlip(0.5),
                        #T.RandomCrop(pretrained_size, padding=10),
                        T.Normalize(mean=pretrained_means,
                                        std=pretrained_stds)
                       ])

test_transforms = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=pretrained_means,
                                        std=pretrained_stds)
                       ])
# %%

class PVClassificationDataset(Dataset):
    def __init__(self, images, labels, transforms):
        super().__init__()
        self.images = images 
        self.labels = labels
        self.transforms = transforms
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = f"../bdappv/ign/img/{image_name}"
        image = np.array(Image.open(image_path).convert("RGB"))
        image = self.transforms(image)
        image_label = torch.tensor(self.labels[idx])
        return image, image_label


train_dataset = PVClassificationDataset(train_images, train_labels, train_transforms)
test_dataset = PVClassificationDataset(test_images, test_labels, test_transforms)

print(len(train_dataset))
print(len(test_dataset))

image, label = train_dataset.__getitem__(1)

#%%
device = torch.device("mps")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

model = torchvision.models.resnet50(weights="IMAGENET1K_V1")

model.fc = nn.Sequential(nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 2))

model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#%%
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"Epoch: {epoch + 1}")
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Step {batch_idx}: loss.item()")

# %%
