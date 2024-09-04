import torch
from utils import Model, preprocessing_mri
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
import random
import numpy as np
import copy


class MedDataset(Dataset):
    def __init__(self, text, img_path, mask_path, transform, type):
        super().__init__()
        self.img_path = img_path
        self.mask_path = mask_path
        self.type = type
        self.f = text
        self.transform = transform

    def __len__(self):
        return len(self.f)

    def __getitem__(self, index):
        img_num = str(self.f.iloc[index, 3])
        label = self.f.iloc[index, 13]
        for file in os.listdir(self.img_path):
            if "_" + img_num + "_" in file:
                img_path = os.path.join(self.img_path, file)
                mask_path = os.path.join(self.mask_path, file)
                for file1 in os.listdir(img_path):
                    if file1 == self.type + ".nii.gz":
                        mri_path = os.path.join(img_path, file1)
                        mask_path = os.path.join(mask_path, file1)
        img_tensor = self.transform(mri_path, mask_path, self.type)
        return img_tensor, label


class main_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model(320, 30, 32, 3, 2, 1024, 4, 16, 2048)

    def forward(self, img_tensor):
        return self.model(img_tensor)


def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(30)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")


img_path = "/Users/gunneo/Documents/Dataset/370/image-before/"
roi_path = "/Users/gunneo/Documents/Dataset/370/ROI/"
train_set = pd.read_csv("/Users/gunneo/Documents/Dataset/370/hnc_train.csv")
val_set = pd.read_csv("/Users/gunneo/Documents/Dataset/370/hnc_valid.csv")
test_set = pd.read_csv("/Users/gunneo/Documents/Dataset/370/hnc_test.csv")
batch_size = 10
shuffle = True
m = main_model().to(device)
m.eval()
train_ds = MedDataset(train_set, img_path, roi_path,
                      preprocessing_mri, "T1+C COR")
val_ds = MedDataset(val_set, img_path, roi_path,
                    preprocessing_mri, "T1+C COR")
test_ds = MedDataset(test_set, img_path, roi_path,
                     preprocessing_mri, "T1+C COR")
dataloader_train = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
dataloader_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
best_auc = 0.0
best_model_wts = copy.deepcopy(m.state_dict())
best_epoch = -1
early_stopping_counter = 0
patience = 20
epochs = 1
losses = []
for epoch in range(epochs):
    iteration = 0
    m.train()
    for img_tensor, labels in dataloader_train:
        img_tensor, labels = img_tensor.to(device), labels.to(device)
        out = m(img_tensor)
        loss = loss_function(out, labels)
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iteration += 1
    print(f"    epoch {epoch + 1} finished.")

    m.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for img_tensor, label in dataloader_val:
            img_tensor, label = img_tensor.to(device), label.to(device)
            out = m(img_tensor)
            pred = torch.argmax(out, dim=1)
            predictions.extend(pred.detach().cpu().numpy())
            labels.extend(label.detach().cpu().numpy())
    print("    ", predictions, labels)
    auc = roc_auc_score(labels, predictions)
    if auc >= best_auc:
        best_auc = auc
        best_epoch = epoch
        best_model_wts = copy.deepcopy(m.state_dict())
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
    if early_stopping_counter >= patience:
        print(
            f"    No improvement in AUROC for {patience} consecutive epochs, stopping early.")
        break
    print(f"    Epoch {epoch + 1}/{epochs} - Validation AUROC: {auc}")
    torch.save(m.state_dict(), 'model_weights_' + str(best_epoch) + '.pth')