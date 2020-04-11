import data.cifar10 as cifar10
import data.nus_wide as nuswide
import data.nus_wide_dual as nuswide_dual
import data.coco_dual as coco_dual
import data.cifar10_dual as cifar10_dual
import data.imagenet as imagenet
import data.coco as coco

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(dataset, root, num_query, num_train, batch_size, num_workers):
    """
    Load dataset.

    Args
        dataset(str): Dataset name.
        root(str): Path of dataset.
        num_query(int): Number of query data points.
        num_train(int): Number of training data points.
        num_workers(int): Number of loading data threads.

    Returns
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
    """
    if dataset == 'cifar-10':
        train_dataloader, query_dataloader, retrieval_dataloader = cifar10.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'nuswide':
        train_dataloader, query_dataloader, retrieval_dataloader = nuswide.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'cifar10_dual':
        train_dataloader, query_dataloader, retrieval_dataloader = cifar10_dual.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'nuswide_dual':
        train_dataloader, query_dataloader, retrieval_dataloader = nuswide_dual.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'coco_dual':
        train_dataloader, query_dataloader, retrieval_dataloader = coco_dual.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'coco':
        train_dataloader, query_dataloader, retrieval_dataloader = coco.load_data(root,
                                                                                     num_query,
                                                                                     num_train,
                                                                                     batch_size,
                                                                                     num_workers,
                                                                                     )
    elif dataset == 'imagenet':
        train_dataloader, query_dataloader, retrieval_dataloader = imagenet.load_data(root,
                                                                                      batch_size,
                                                                                      num_workers,
                                                                                      )
    else:
        raise ValueError("Invalid dataset name!")

    return train_dataloader, query_dataloader, retrieval_dataloader
