# -*- coding:utf-8 -*-
import numpy as np
import os
import pdb

def mean_average_precision2(database_code, validation_code, database_labels, validation_labels):
    # all four variables are numpy array
    R = 5000
    query_num = validation_code.shape[0]
    sim = np.dot(database_code, validation_code.T)
    ids = np.argsort(-sim, axis=0)
    APx = []
    ground_truth = np.sum(np.dot(database_labels, validation_labels.T), axis=0)
    validation_labels[validation_labels == 0] = -1
    for i in range(query_num):
        label = validation_labels[i, :]
        if ground_truth[i] > 0:
            idx = ids[:, i]
            imatch = np.sum(database_labels[idx[0:R], :] == label, axis=1) > 0
            relevant_num = np.sum(imatch)
            Lx = np.cumsum(imatch)
            Px = Lx.astype(float) / np.arange(1, R + 1, 1)
            if relevant_num != 0:
                APx.append(np.sum(Px * imatch) / relevant_num)
            else:
                APx.append(np.sum(Px * imatch) / 1)
        #else:
        #    APx.append(0)
    return np.mean(np.array(APx))


def compact_label(x):
    label = np.zeros([x.shape[0]])
    for i in range(x.shape[0]):
        y = x[i, ...]
        for ind in range(len(y)):
            if y[ind] == 1:
                label[i] = ind
    return label

def sparse_label(x):
    label = np.zeros([x.shape[0],10])
    for i in range(x.shape[0]):
        label[i,x[i]] = 1
    return label


