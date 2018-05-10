# -*- coding: utf-8 -*-

import csv
import os

import numpy as np
import nsml
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from nsml import DATASET_PATH, DATASET_NAME
from nsml import GPU_NUM
IS_DATASET = True
import argparse

csv.field_size_limit(500 * 1024 * 1024 * 10000)
train_data_name = 'train_data'
train_label_name = 'train_label'
test_data_name = 'test_data'

def cusmtom_collate_fn(data):
    """Creates mini-batch tensors from the list of list [batch_input, batch_target].
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of list [batch_input, batch_target].

    Refs:
        Thanks to yunjey. (https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py)
    """

    # data sampler returns an indices to select into mini-batch.
    # than the collate_function gather the selected data.
    # in this example, we use pack_padded_sequence so always ordering with decsending
    # so the targets should be ordered samely here.

    # sort the sequences as descending order to use 'pack_padded_sequence' before loading.
    data.sort(key=lambda x: len(x[0]), reverse=True)
    tensor_targets = torch.LongTensor([pair[1] for pair in data])
    return [data, tensor_targets]

def data_loader(dataset_path, train=False, batch_size=200, ratio_of_validation=0.1, shuffle=True):
    if train:
        return get_dataloaders(dataset_path=dataset_path, type_of_data='train',
                               batch_size=batch_size, ratio_of_validation=ratio_of_validation,
                               shuffle=shuffle)
    else:
        data_dict = {'data': read_test_file(dataset_path=DATASET_PATH)}
        return data_dict

def get_dataloaders(dataset_path, type_of_data, batch_size, ratio_of_validation=0.2, shuffle=True):
    if type_of_data == 'train':
        num_workers = 0  # num of threads to load data, default is 0. if you use thread(>1), don't confuse even if debug messages are reported asynchronously.
        train_movie_dataset = MR_DataSet(dataset_path=dataset_path)

        num_train = len(train_movie_dataset)
        split_point = int(ratio_of_validation*num_train)

        indices = list(range(num_train))
        if shuffle:
            np.random.shuffle(indices)

        train_idx, val_idx = indices[split_point:], indices[: split_point]
        train_sampler = SubsetRandomSampler(train_idx)  # Random sampling at every epoch without replacement in given indices
        val_sampler = SubsetRandomSampler(val_idx)

        # 'pin_memory=True' allows for you to use fast memory buffer with way of calling '.cuda(async=True)' function.
        pin_memory = True if GPU_NUM else False

        train_loader = torch.utils.data.DataLoader(
            train_movie_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers,
            pin_memory=pin_memory, collate_fn=cusmtom_collate_fn)
        val_loader = torch.utils.data.DataLoader(
            train_movie_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers,
            pin_memory=pin_memory, collate_fn=cusmtom_collate_fn)
        return [train_loader, val_loader]

    else:
        raise BaseException('This function is only for train data set')


def read_test_file(dataset_path):
    data_path = os.path.join(dataset_path, 'test', test_data_name)  # Test data only
    loaded_data = []
    with open(data_path, 'rt') as data_csvfile:
        read_line = csv.reader(data_csvfile)
        for idx, data in enumerate(read_line):
            print(read_line)
            data = data[0]
            loaded_data.append([data])
    return loaded_data

class MR_DataSet(Dataset):
    def __init__(self, dataset_path):
        data_path = os.path.join(dataset_path, 'train', train_data_name)
        data_label = os.path.join(dataset_path, 'train', train_label_name)

        print("Data path: ", data_path)
        self.loaded_data = []

        with open(data_path, 'rt', 50 * 1024 * 1024) as data_csvfile,\
                open(data_label, 'rt') as label_csv:
            #self.total_data_length = sum(1 for _ in data_csvfile)
            #print("Total Data Length : ", self.total_data_length)

            # Reset File pointer to 0 once it has completed counting file lines
            data_csvfile.seek(0)
            label_csv.seek(0)
            read_line = csv.reader(data_csvfile)
            read_label = csv.reader(label_csv)

            print("TEXT : ", read_line, "    LABEL : ", read_label)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--epochs', type=int, default=1)  # 300
    args.add_argument('--batch', type=int, default=1000)  # 1000
    args.add_argument('--embedding', type=int, default=8)  # 8
    args.add_argument('--hidden', type=int, default=512)  # 512
    args.add_argument('--layers', type=int, default=2)  # 2
    args.add_argument('--initial_lr', type=float, default=0.001)  # default : 0.001 (initial learning rate)
    args.add_argument('--char', type=int, default=251)  # Do not change this
    args.add_argument('--output', type=int, default=1)  # Do not change this
    args.add_argument('--mode', type=str, default='train')  # 'train' or 'test' (for nsml)
    args.add_argument('--pause', type=int, default=0)  # Do not change this (for nsml)
    args.add_argument('--iteration', type=str, default='0')  # Do not change this (for nsml)
    config = args.parse_args()

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        if IS_DATASET:
            train_loader, val_loader = data_loader(dataset_path=DATASET_PATH, train=True,
                                                   batch_size=config.batch,
                                                   ratio_of_validation=0.1, shuffle=True)
        else:
            data_path = '../dummy/movie_review/'  # NOTE: load from local PC
            train_loader, val_loader = data_loader(dataset_path=data_path, train=True,
                                                   batch_size=config.batch,
                                                   ratio_of_validation=0.1, shuffle=True)