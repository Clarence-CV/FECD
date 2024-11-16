import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F
from copy import deepcopy

# from utils import *
# from resnet import *
from torch.autograd import Variable
import torch
import torch.nn as nn

from torch.utils.data import Dataset

from PIL import Image



class RafDataset(data.Dataset):
    def __init__(self, phase, dataset_name, basic_aug=True, transform=None):
        self.raf_path = './FERdata/'
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        if dataset_name == "rafdb":
            df = pd.read_csv('./FERdata/list_patition_label.txt', sep=' ', header=None)
            name_c = 0
            label_c = 1

            if phase == 'train':
                dataset = df[df[name_c].str.startswith('train')]
            else:
                dataset = df[df[name_c].str.startswith('test')]

            self.targets = dataset.iloc[:, label_c].values
            images_names = dataset.iloc[:, name_c].values
            self.uq_idxs = np.array(range(len(self.targets)))
            self.file_paths = []#data

            for f in images_names:
                f = f.split(".")[0]
                f += '_aligned.jpg'
                file_name = os.path.join(self.raf_path, 'aligned', f)
                self.file_paths.append(file_name)

        elif dataset_name == "ferplus":
            df = pd.read_csv('./FERdata/Ferplus/ferplus_labels.csv')
            name_c = "Image"
            
            label_c = "Emotion"

            if phase == 'train':
                dataset = df[df["Type"] == "train"]
            else:
                dataset = df[df["Type"] == "test"]

            self.targets = dataset[label_c].values
            images_names = dataset[name_c].values
            
            self.uq_idxs = np.array(range(len(self.targets)))
            self.file_paths = []#data

            for f in images_names:
                file_name = os.path.join(self.raf_path, 'Ferplus/train/', f)
                self.file_paths.append(file_name)
            
    def __len__(self):
        return len(self.file_paths)
    
    def get_labels(self):
        return self.label
    
    def __getitem__(self, idx):
        target = self.targets[idx]
        image = cv2.imread(self.file_paths[idx])
        uq_idx = self.uq_idxs[idx]
        image = image[:, :, ::-1]
        img = Image.fromarray(image)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, uq_idx

    
class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])
    
    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)
    

def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices

def subsample_dataset(dataset, idxs):

    # Allow for setting in which all empty set of indices is passed

    if len(idxs) > 0:

        dataset.file_paths = [dataset.file_paths[idx] for idx in idxs]
#         dataset.data = dataset.data[idxs]
        dataset.targets = np.array(dataset.targets)[idxs].tolist()
        dataset.uq_idxs = dataset.uq_idxs[idxs]

        return dataset

    else:
        return None
    
def subsample_classes(dataset, include_classes):


    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    dataset = subsample_dataset(dataset, cls_idxs)


    return dataset

def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs

def get_rafdb_datasets(dataset_name, train_transform, test_transform, train_classes,
                       prop_train_labels=0.8, split_train_val=False, seed=0):

    np.random.seed(seed)
    print(train_classes)

    # Init entire training set
    whole_training_set = RafDataset(phase='train', dataset_name = dataset_name, transform=train_transform)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    train_dataset_labelled = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)
    subsample_indices = subsample_instances(train_dataset_labelled, prop_indices_to_subsample=prop_train_labels)
    train_dataset_labelled = subsample_dataset(train_dataset_labelled, subsample_indices)

    # Split into training and validation sets
    train_idxs, val_idxs = get_train_val_indices(train_dataset_labelled)
    train_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), train_idxs)
    val_dataset_labelled_split = subsample_dataset(deepcopy(train_dataset_labelled), val_idxs)
    val_dataset_labelled_split.transform = test_transform

    # Get unlabelled data
    unlabelled_indices = set(whole_training_set.uq_idxs) - set(train_dataset_labelled.uq_idxs)
    train_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set), np.array(list(unlabelled_indices)))

    # Get test set for all classes
    test_dataset = RafDataset(phase='test', dataset_name = dataset_name, transform=test_transform)

    # Either split train into train and val or use test set as val
    train_dataset_labelled = train_dataset_labelled_split if split_train_val else train_dataset_labelled
    val_dataset_labelled = val_dataset_labelled_split if split_train_val else None

    all_datasets = {
        'train_labelled': train_dataset_labelled,
        'train_unlabelled': train_dataset_unlabelled,
        'val': val_dataset_labelled,
        'test': test_dataset,
    }

    return all_datasets

def get_RAFDB_datasets(dataset_name, train_transform, test_transform, args):

    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """


    # Get datasets
    datasets =  get_rafdb_datasets(dataset_name = dataset_name, train_transform=train_transform, test_transform=test_transform,train_classes=args.train_classes,prop_train_labels=args.prop_train_labels,
split_train_val=False)
    
    # Set target transforms:
    target_transform_dict = {}
    for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            dataset.target_transform = target_transform

    # Train split (labelled and unlabelled classes) for training
    train_dataset = MergedDataset(labelled_dataset=deepcopy(datasets['train_labelled']),
                                  unlabelled_dataset=deepcopy(datasets['train_unlabelled']))#

    test_dataset = datasets['test']
    unlabelled_train_examples_test = deepcopy(datasets['train_unlabelled'])
    unlabelled_train_examples_test.transform = test_transform

    return train_dataset, test_dataset, unlabelled_train_examples_test, datasets

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        if not isinstance(self.base_transform, list):
            return [self.base_transform(x) for i in range(self.n_views)]
        else:
            return [self.base_transform[i](x) for i in range(self.n_views)]


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_fer_data(data_path="SimGCD/data_embed_npy.npy",
                 label_path="SimGCD/label_npu.npy"):

    data = np.load(data_path)
    label = np.load(label_path)
    n_samples, n_features = data.shape

    return data, label, n_samples, n_features

