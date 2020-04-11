import torch
import torch.optim as optim
import os
import time
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.model_loader import load_model
from loguru import logger
from models.dsdh_loss import DDDHLoss
from utils.evaluate import mean_average_precision
from utils.dataprocess import mean_average_precision2

import pdb

def train(
    train_dataloader,
    query_dataloader,
    retrieval_dataloader,
    arch,
    code_length,
    label_length,
    device,
    lr,
    max_iter,
    nu,
    eta,
    lamda,
    weight_decay,
    topk,
    evaluate_interval,
 ):
    """
    Training model.

    Args
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.DataLoader): Data loader.
        arch(str): CNN model name.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter: int
        Maximum iteration
        nu, eta(float): Hyper-parameters.
        topk(int): Compute mAP using top k retrieval result
        evaluate_interval(int): Evaluation interval.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Construct network, optimizer, loss
    model = load_model(arch, code_length, label_length).to(device)
    criterion = DDDHLoss(nu, eta)
    params_to_filter = list(map(id, model.hash_layer.parameters())) + list(map(id, model.classifier.parameters()))
    last_layers_params = list(model.hash_layer.parameters()) + list(model.classifier.parameters())
    base_params = filter(lambda p: id(p) not in params_to_filter, model.parameters())
    optimizer = optim.Adam([
        {'params': base_params},
        {'params': last_layers_params, 'lr': 10*lr, 'weight_decay': lamda}],
            lr = lr,
            weight_decay=weight_decay,
            )
    scheduler = CosineAnnealingLR(optimizer, max_iter, 1e-7)

    # Initialize
    best_map = 0.
    best_epoch = 0
    iter_time = time.time()
    loss_all = 0
    batch_cnt = 0

    for it in range(max_iter):
        model.train()
        # CNN-step
        for data1, targets1, data2, targets2, index in train_dataloader:
            data1, targets1, data2, targets2 = data1.to(device), targets1.to(device), data2.to(device), targets2.to(device)
            y1 = targets1.float() # batch_size, label_dim
            y2 = targets2.float() # batch_size, label_dim
            S = targets1.float() @ targets2.float().t() > 0 # batch_size, batch_size
            optimizer.zero_grad()

            f1, g1 = model(data1) # batch_size, code_len; batch_size, label_dim
            f2, g2 = model(data2) # batch_size, code_len; batch_size, label_dim
            f1, f2, g1, g2 = f1.float(), f2.float(), g1.float(), g2.float()

            loss = criterion(f1, f2, S, y1, y2, g1, g2 )
            loss_all += loss.cpu().data
            batch_cnt += 1
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Evaluate
        if it % evaluate_interval == evaluate_interval - 1:
            iter_time = time.time() - iter_time

            # Generate hash code
            query_code = generate_code(model, query_dataloader, code_length, device)
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)
            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

            # Compute map
            #mAP = mean_average_precision(
            #    query_code.to(device),
            #    retrieval_code.to(device),
            #    query_targets.to(device),
            #    retrieval_targets.to(device),
            #    device,
            #    topk,
            #)
            mAP = mean_average_precision2(
                    retrieval_code.numpy(),
                    query_code.numpy(),
                    retrieval_targets.numpy(),
                    query_targets.numpy()
                    )
            loss_print = loss_all/batch_cnt
            logger.info('[iter:{}/{}][loss:{:.4f}][map:{:.4f}][time:{:.2f}]'.format(it+1, max_iter, loss_print, mAP, iter_time))

            # Save checkpoint
            if best_map < mAP:
                best_map = mAP
                best_epoch = it
                checkpoint = {
                    'qB': query_code,
                    'qL': query_targets,
                    'rB': retrieval_code,
                    'rL': retrieval_targets,
                    'model': model.state_dict(),
                    'map': best_map,
                }
            iter_time = time.time()
        # early stop
        if it - 80 > best_epoch:
            break

    return checkpoint


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor, n*code_length): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code,_ = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code
