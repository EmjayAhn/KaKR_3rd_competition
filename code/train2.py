import os
import copy
import time
import random
from tqdm import tqdm

import PIL
from PIL import Image


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torchvision
from torchvision import datasets, transforms, models

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import f1_score
# fix seeds
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

SEED = 2019
seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = False
print(device)

df_class = pd.read_csv('../data/class.csv')
df_train = pd.read_csv('../data/train.csv')
df_train = df_train[['img_file', 'class']]
df_train.replace(196, 0, inplace=True)

X_train, X_val, y_train, y_val = train_test_split(df_train['img_file'], df_train['class'], stratify=df_train['class'], test_size=0.2, random_state=SEED)

X_train = X_train.values
X_val = X_val.values
y_train = y_train.values
y_val = y_val.values

TRAIN_DATA_PATH = '../data/train_crop/'
TEST_DATA_PATH = '../data/test_crop/'

class TrainImages(Dataset):
    def __init__(self, images, labels, mode=None, transforms=None):
        self.images = images
        self.labels = labels
        self.mode = mode
        self.transforms = transforms[self.mode]
        
    def __len__(self):
        return self.images.shape[0]
        
    def __getitem__(self, idx):
        image = Image.open(TRAIN_DATA_PATH + self.images[idx]).convert("RGB")
        image = self.transforms(image)
        label = self.labels[idx]
        
        return image, label
    
    
class TestImages(Dataset):
    def __init__(self, images, labels, mode=None, transforms=None):
        self.images = images
        self.laels = labels
        self.mode = mode
        self.transforms = transforms[self.mode]
        
    def __getitem__(self, idx):
        image = Image.open(TEST_DATA_PATH + self.images[idx]).convert("RGB")
        image = self.transforms(image)
        labels = self.labels[idx]
        
        return image, label

transform = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])
}

batch_size = 64

train_dataset = TrainImages(images=X_train, labels=y_train, mode='train', transforms=transform)
val_dataset = TrainImages(images=X_val, labels=y_val, mode='val', transforms=transform)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders = {
    'train': train_dataloader,
    'val': val_dataloader
}

dataset_sizes = {
    'train': len(train_dataset),
    'val': len(val_dataset)
}

model_res = models.resnet101(pretrained=True, progress=False)
num_features = model_res.fc.in_features
model_res.fc = nn.Linear(num_features, 196)
model_res.load_state_dict(torch.load('../model/ten_crop/rough/best_model_5_101.pt'))

optimizer = optim.Adam(model_res.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, device, PATH, epochs=20):
    start = time.time()

    num_classes = 196

    best_model_weights = copy.deepcopy(model.state_dict())
    best_f1 = 0.0

    for epoch in tqdm(range(epochs)):
        print("EPOCH {} / {}: ".format(epoch+1, epochs))
        print("-" * 10)

        epoch_loss = 0.0
        phase = 'train'

        for batch_index, (batch_inputs, batch_labels) in enumerate(dataloaders[phase]):
            batch_inputs = batch_inputs.cuda()
            batch_labels = batch_labels.cuda()

            optimizer.zero_grad()
            outputs = model(batch_inputs)
#             _, preds = torch.max(outputs, 1)
            batch_loss = criterion(outputs, batch_labels)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += batch_loss.item() * batch_inputs.size(0)
            if batch_index % 5 == 0:
                print("EPOCH {} BATCH {}: training batch loss: {}".format(epoch+1, batch_index+1, batch_loss.item()))

            if batch_index % 10 == 0:
                phase = 'val'
                val_preds = np.zeros((dataset_sizes['val'], 1))
                val_loss = 0.0
                with torch.no_grad():
                    model.eval()
                    for val_batch_index, (val_batch_inputs, val_batch_labels) in enumerate(dataloaders[phase]):

                        val_batch_inputs = val_batch_inputs.cuda()
                        val_batch_labels = val_batch_labels.cuda()

                        val_outputs = model(val_batch_inputs).detach()
                        _, val_batch_preds = torch.max(val_outputs, 1)
                        val_batch_loss = criterion(val_outputs, val_batch_labels)
                        val_preds[val_batch_index * batch_size: (val_batch_index+1) * batch_size] = val_batch_preds.cpu().view(-1, 1).numpy()
                        val_loss += val_batch_loss.item() * val_batch_inputs.size(0)

                    val_score = f1_score(y_val, val_preds, average='micro')
                    print()
                    print(">>>>>>  EPOCH {} BATCH {}: validation score {}".format(epoch+1, batch_index+1, val_score))
                    print()
                    if val_score > best_f1:
                        best_f1 = val_score
                        best_model_weights = copy.deepcopy(model.state_dict())
                        torch.save(model.state_dict(), '../{}/best_model_{}_{}.pt'.format(PATH, epoch+1, batch_index+1))

                phase = 'train'
                model.train()

        epoch_loss = epoch_loss / dataset_sizes['train']
        print("EPOCH {}: EPOCH_LOSS: {}".format(epoch+1, epoch_loss))
    end = time.time()
    elapsed_time = end - start
    print("Training COMPLETED: {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("BEST VALIDATION F1: {:4f}".format(best_f1))

    model.load_state_dict(best_model_weights)
    return model

model_res.to(device)
model_res = train_model(model=model_res, dataloaders=dataloaders, dataset_sizes=dataset_sizes, criterion=criterion, optimizer=optimizer, device=device, epochs=20, 
        PATH='model/ten_crop/tune')



# model_res.to(device)
# optimizer = optim.Adam(model_res.parameters(), lr=0.000001)
# criterion = nn.CrossEntropyLoss()
# model_res = train_model(model=model_res, dataloaders=dataloaders, dataset_sizes=dataset_sizes, criterion=criterion, optimizer=optimizer, device=device, epochs=100, PATH='model/ten_crop/fine_tune')

