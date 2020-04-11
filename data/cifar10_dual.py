import os
import pickle
import sys
import pdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform_dddh import train_transform, query_transform, Onehot, encode_onehot


def load_data(root, num_query, num_train, batch_size, num_workers):
    """
    Load cifar10 dataset.

    Args
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader(torch.evaluate.data.DataLoader): Data loader.
    """
    all_data, QUERY_IMG, QUERY_TARGET, TRAIN_IMG, TRAIN_IMG2, TRAIN_TARGET, TRAIN_TARGET2, RETRIEVAL_IMG, RETRIEVAL_TARGET = init_data(root, num_query, num_train)

    query_dataset = CIFAR10_test(
            QUERY_IMG, QUERY_TARGET, transform=query_transform(), target_transform=Onehot())

    train_dataset = CIFAR10_train(
            TRAIN_IMG, TRAIN_TARGET,TRAIN_IMG2, TRAIN_TARGET2, transform=train_transform(), target_transform=Onehot())

    retrieval_dataset = CIFAR10_test(
            RETRIEVAL_IMG, RETRIEVAL_TARGET,
            transform=query_transform(), target_transform=Onehot())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=num_workers,
      )
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
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

def init_data(root, num_query, num_train):
    data_list = ['data_batch_1',
                 'data_batch_2',
                 'data_batch_3',
                 'data_batch_4',
                 'data_batch_5',
                 'test_batch',
                 ]
    base_folder = 'cifar-10-batches-py'

    data = []
    targets = []

    for file_name in data_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    targets = np.array(targets)

    # Sort by class
    sort_index = targets.argsort()
    data = data[sort_index, :]
    targets = targets[sort_index]

    # (num_query / number of class) query images per class
    # (num_train / number of class) train images per class
    query_per_class = num_query // 10
    train_per_class = num_train // 10

    # Permutate index (range 0 - 6000 per class)
    perm_index = np.random.permutation(data.shape[0] // 10)
    query_index = perm_index[:query_per_class]
    train_index_ = perm_index[query_per_class: query_per_class + train_per_class]

    query_index = np.tile(query_index, 10)
    train_index = np.tile(train_index_, 10)
    np.random.shuffle(train_index_)
    train_index2 = np.tile(train_index_, 10)
    inc_index = np.array([i * (data.shape[0] // 10) for i in range(10)])
    query_index = query_index + inc_index.repeat(query_per_class)
    train_index = train_index + inc_index.repeat(train_per_class)
    train_index2 = train_index2 + inc_index.repeat(train_per_class)
    list_query_index = [i for i in query_index]
    retrieval_index = np.array(list(set(range(data.shape[0])) - set(list_query_index)), dtype=np.int)

    # Split data, targets
    QUERY_IMG = data[query_index, :]
    QUERY_TARGET = targets[query_index]
    TRAIN_IMG = data[train_index, :]
    TRAIN_IMG2 = data[train_index2, :]
    TRAIN_TARGET = targets[train_index]
    TRAIN_TARGET2 = targets[train_index2]
    RETRIEVAL_IMG = data[retrieval_index, :]
    RETRIEVAL_TARGET = targets[retrieval_index]

    return data, QUERY_IMG, QUERY_TARGET, TRAIN_IMG, TRAIN_IMG2, TRAIN_TARGET, TRAIN_TARGET2, RETRIEVAL_IMG, RETRIEVAL_TARGET

class CIFAR10_train(Dataset):
    """
    Cifar10 dataset.
    """

    def __init__(self, TRAIN_IMG, TRAIN_TARGET, TRAIN_IMG2, TRAIN_TARGET2, transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        self.data = TRAIN_IMG
        self.targets = TRAIN_TARGET
        self.data2 = TRAIN_IMG2
        self.targets2 = TRAIN_TARGET2

        self.onehot_targets = encode_onehot(self.targets, 10)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img2, target2 = self.data2[index], self.targets2[index]

        img2 = Image.fromarray(img2)
        if self.transform is not None:
            img2 = self.transform(img2)
        if self.target_transform is not None:
            target2 = self.target_transform(target2)
        #img2 = Image.fromarray(img2)

        return img, target, img2, target2, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.from_numpy(self.onehot_targets).float()

class CIFAR10_test(Dataset):
    """
    Cifar10 dataset.
    """

    def __init__(self, IMG, TARGET,
                 transform=None, target_transform=None,
                 ):
        self.transform = transform
        self.target_transform = target_transform

        self.data = IMG
        self.targets = TARGET

        self.onehot_targets = encode_onehot(self.targets, 10)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, index) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        """
        Return one-hot encoding targets.
        """
        return torch.from_numpy(self.onehot_targets).float()
