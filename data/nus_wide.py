import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(tc, root, num_query, num_train, batch_size, num_workers,
):
    """
    Loading nus-wide dataset.

    Args:
        tc(int): Top class.
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """

    query_dataset = NusWideDatasetTC81(
        root,
        'test.txt',
        transform=query_transform(),
    )

    train_dataset = NusWideDatasetTC81(
        root,
        'database.txt',
        transform=train_transform(),
        train=True,
        num_train=num_train,
    )

    retrieval_dataset = NusWideDatasetTC81(
        root,
        'database_img.txt',
        transform=query_transform(),
    )

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader


class NusWideDatasetTC81(Dataset):
    """
    Nus-wide dataset, 81 classes.

    Args
        root(str): Path of image files.
        img_txt(str): Path of txt file containing image file name.
        label_txt(str): Path of txt file containing image label.
        transform(callable, optional): Transform images.
        train(bool, optional): Return training dataset.
        num_train(int, optional): Number of training data.
    """
    def __init__(self, root, data_txt, transform=None, train=None, num_train=None):
        self.root = root
        self.transform = transform

        data_txt_path = os.path.join(root, data_txt)


        # Read files
        img = []
        target = []
        with open(data_txt_path, 'r') as f:
            for i in f:
                line = i.rstrip('\n').split('')
                img.append(line[0])
                target.append(line[1:])
        self.img = np.array(img)
        self.targets = np.array(target)


        # Sample training dataset
        if train is True:
            perm_index = np.random.permutation(len(self.img))[:num_train]
            self.img = self.img[perm_index]
            self.targets = self.targets[perm_index]

    def __getitem__(self, index):

        img = Image.open(os.path.join(self.root, self.img[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index], index

    def __len__(self):
        return len(self.img)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()