import os
import pdb
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_query, num_train, batch_size, num_workers,
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

    query_dataset = coco_dual(
        root,
        'coco/test/test.txt',
        transform=query_transform(),
    )

    train_dataset = coco_train(
        root,
        'coco/train.txt',
        transform=train_transform(),
        train=True,
        num_train=num_train,
    )

    retrieval_dataset = coco_dual(
        root,
        'coco/test/database0.txt',
        transform=query_transform(),
    )

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader


class coco_dual(Dataset):
    def __init__(self, root, data_txt, transform=None, train=None, num_train=None):
        self.root = root
        self.transform = transform

        data_txt_path = os.path.join(root, data_txt)

        # Read files
        img = []
        target = []
        with open(data_txt_path, 'r') as f:
            for i in f:
                line = i.rstrip('\n').split(' ')
                img.append(line[0][7:])
                target.append([int(j) for j in line[1:]])
        self.img = np.array(img)
        self.targets = np.array(target)


    def __getitem__(self, index):

        img = Image.open(os.path.join(self.root, self.img[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[index], index

    def __len__(self):
        return len(self.img)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

class coco_train(Dataset):
    def __init__(self, root, data_txt, transform=None, train=None, num_train=None):
        self.root = root
        self.transform = transform

        data_txt_path = os.path.join(root, data_txt)

        # Read files
        img = []
        target = []
        with open(data_txt_path, 'r') as f:
            for i in f:
                line = i.rstrip('\n').split(' ')
                img.append(line[0][7:])
                target.append([int(j) for j in line[1:]])
        self.img = np.array(img)
        self.targets = np.array(target)


        # Sample training dataset
        if train is True:
            perm_index = np.random.permutation(len(self.img))[:num_train]
            self.img1 = self.img[perm_index]
            self.targets1 = self.targets[perm_index]

            np.random.shuffle(perm_index)
            self.img2 = self.img[perm_index]
            self.targets2 = self.targets[perm_index]

    def __getitem__(self, index):

        img1 = Image.open(os.path.join(self.root, self.img1[index])).convert('RGB')
        img2 = Image.open(os.path.join(self.root, self.img2[index])).convert('RGB')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, self.targets1[index], img2, self.targets2[index], index

    def __len__(self):
        return len(self.img1)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()
