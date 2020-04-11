import torch
import numpy as np
import os
import random
import argparse
import dddh

from loguru import logger
from data.data_loader import load_data


def run():
    args = load_config()
    logger.add(
        os.path.join('/data4/binyi/DSDH_data/logs', '{}_model_{}_codelength_{}__nu_{}_eta_{}_topk_{}.log'.format(
            args.dataset,
            args.arch,
            ','.join([str(c) for c in args.code_length]),
            args.nu,
            args.eta,
            args.topk,
            )),
        rotation='500 MB',
        level='INFO',
    )
    logger.info(args)

    torch.backends.cudnn.benchmark = True
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    query_dataloader, train_dataloader, retrieval_dataloader = load_data(
        args.dataset,
        args.root,
        args.num_query,
        args.num_train,
        args.batch_size,
        args.num_workers,
    )

    # Training
    for code_length in args.code_length:
        logger.info('[code length:{}]'.format(code_length))
        checkpoint = dddh.train(
            train_dataloader,
            query_dataloader,
            retrieval_dataloader,
            args.arch,
            code_length,
            args.label_length,
            args.device,
            args.lr,
            args.max_iter,
            args.nu,
            args.eta,
            args.lamda,
            args.weight_decay,
            args.topk,
            args.evaluate_interval,
        )

        # Save checkpoint
        torch.save(checkpoint, os.path.join('/data4/binyi/DSDH_data/checkpoints', '{}_model_{}_codelength_{}__nu_{}_eta_{}_topk_{}_map_{:.4f}.pt'.format(args.dataset, args.arch, code_length, args.nu, args.eta, args.topk, checkpoint['map'])))
        logger.info('[code_length:{}][map:{:.4f}]'.format(code_length, checkpoint['map']))


def load_config():
    """
    Load configuration.

    Args
        None

    Returns
        args(argparse.ArgumentParser): Configuration.
    """
    parser = argparse.ArgumentParser(description='DDDH_PyTorch')
    parser.add_argument('--dataset', default='nuswide_dual',
                        help='Dataset name.')
    parser.add_argument('--root', default='/data4/binyi/DSDH_data/',
                        help='Path of dataset')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size.(default: 128)')
    parser.add_argument('--arch', default='alexnet_dddh', type=str,
                        help='CNN model name.(default: alexnet)')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate.(default: 1e-5)')
    parser.add_argument('--code-length', default='24,32,48,64', type=str,
                        help='Binary hash code length.(default: 12,24,32,48)')
    parser.add_argument('--max-iter', default=400, type=int,
                        help='Number of iterations.(default: 150)')
    parser.add_argument('--num-query', default=1000, type=int,
                        help='Number of query data points.(default: 1000)')
    parser.add_argument('--num-train', default=5000, type=int,
                        help='Number of training data points.(default: 5000)')
    parser.add_argument('--num-workers', default=6, type=int,
                        help='Number of loading data threads.(default: 6)')
    parser.add_argument('--topk', default=5000, type=int,
                        help='Calculate map of top k.(default: all)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='Using gpu.(default: False)')
    parser.add_argument('--nu', default=5, type=float,
                        help='Hyper-parameter.(default: 1)')
    parser.add_argument('--eta', default=0.5, type=float,
                        help='Hyper-parameter.(default: 1e-2)')
    parser.add_argument('--lamda', default=0.5, type=float,
                        help='Hyper-parameter.(default: 1e-2)')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='Hyper-parameter.(default: 1e-2)')
    parser.add_argument('--evaluate-interval', default=10, type=int,
                        help='Evaluation interval.(default: 10)')
    parser.add_argument('--seed', default=3367, type=int,
                        help='Random seed.(default: 3367)')

    args = parser.parse_args()

    # GPU
    if args.gpu is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device("cuda:%d" % args.gpu)

    # Hash code length
    args.code_length = list(map(int, args.code_length.split(',')))
    if args.dataset in ['nuswide_dual', 'nuswide_81']:
        args.label_length = 81
    elif args.dataset == 'coco_dual':
        args.label_length = 80
    elif args.dataset == 'cifar10_dual':
        args.label_length = 10
    return args


if __name__ == '__main__':
    run()

