from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from attrdict import AttrDict
import os
from sklearn.model_selection import train_test_split
import numpy as np


class TiangongDataset(Dataset):
    def __init__(self, images, labels, transforms=None):
        self.transforms = transforms
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        image = self.transforms(image)
        return image, label


class Metadata:
    def __init__(self, data_path):
        self.encoder = LabelEncoder()
        train_data = pd.read_csv(os.path.join(
            data_path, 'train.csv'), names=['image', 'label'])
        train_data.image = data_path + 'train/' + train_data.image
        train_data.label = self.encoder.fit_transform(train_data.label)
        images_train, images_val, labels_train, labels_val = train_test_split(
            train_data.image.values, train_data.label.values, random_state=12345, test_size=0.3)

        test_data = pd.DataFrame(os.listdir(
            os.path.join(data_path, 'test')), columns=['image'])
        test_data.image = data_path + 'test/' + test_data.image
        images_test = test_data.image.values
        self.data = AttrDict({
            'images_train': images_train,
            'images_val': images_val,
            'labels_train': labels_train,
            'labels_val': labels_val,
            'images_test': images_test,
            'labels_test': np.ones(len(images_test), dtype='int64'),
        })

    def decode(self, labels_test):
        return self.encoder.inverse_transform(labels_test)


def get_fulldataloader(metadata, batch_size=128):
    dataloader = DataLoader(
        TiangongDataset(
            images=np.r_[metadata.data.images_train, metadata.data.images_val],
            labels=np.r_[metadata.data.labels_train, metadata.data.labels_val],
            transforms=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    return dataloader


def get_dataloader(metadata, batch_size=128):
    # train data
    dataloader_train = DataLoader(
        TiangongDataset(
            images=metadata.data.images_train,
            labels=metadata.data.labels_train,
            transforms=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]),
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

# val data
    dataloader_val = DataLoader(
        TiangongDataset(
            images=metadata.data.images_val,
            labels=metadata.data.labels_val,
            transforms=transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
# test data
    dataloader_test = DataLoader(
        TiangongDataset(
            images=metadata.data.images_test,
            labels=metadata.data.labels_test,
            transforms=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return dataloader_train, dataloader_val, dataloader_test
