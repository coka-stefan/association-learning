#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import warnings
from sklearn import preprocessing
from skimage.io import imread
from skimage import transform
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision.datasets import CIFAR10

from torchvision import transforms, models
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# run_name = "resnet18_not_pretrained_6_classes_predict_first_acc_first"
run_name = "object-object-6 classes plane-car-truck-ship"

# Switching on GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: {}".format(device))

writer = SummaryWriter(f'runs/{run_name}')

RC = transforms.RandomCrop(32, padding=4)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
RR = transforms.RandomRotation((-10, 10))
NRM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage()

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([TPIL, RC, RR, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([TT, NRM])

# Downloading/Louding CIFAR10 data
trainset = CIFAR10(root='./CIFAR10', train=True, download=True)
testset = CIFAR10(root='./CIFAR10', train=False, download=True)
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

# print(trainset.classes)
# Separating trainset/testset data/label
x_train = trainset.data
x_test = testset.data
y_train = trainset.targets
y_test = testset.targets


class_label_mapper = {
    0: [0, 4],
    1: [1, 5],
    2: [2, 5],
    3: [3, 4]
}

# Define a function to separate CIFAR classes by class index


def get_class_i(x, y, i):
    """
    x: trainset.train_data or testset.test_data
    y: trainset.train_labels or testset.test_labels
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i


class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of
        list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)

        labels = [class_label_mapper[class_label]]

        return img, ['plane', 'cat', 'frog', 'ship'][class_label], self.to_onehot(labels, 6)
        # return img, trainset.classes[class_label], self.to_onehot(labels, 6)

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls
        in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class

    def to_onehot(self, labels, n_categories, dtype=torch.float):
        one_hot_labels = torch.zeros(size=(n_categories, ), dtype=dtype)
        for label in labels:
            one_hot_labels[label] = 1.
        return one_hot_labels


avian_land_dataset = DatasetMaker(
    [
        get_class_i(x_train, y_train, classDict['plane']),
        get_class_i(x_train, y_train, classDict['car']),
        get_class_i(x_train, y_train, classDict['truck']),
        get_class_i(x_train, y_train, classDict['ship'])
    ],
    transform_with_aug
)


test_avian_land_dataset = DatasetMaker(
    [
        get_class_i(x_train, y_train, classDict['plane']),
        get_class_i(x_train, y_train, classDict['car']),
        get_class_i(x_train, y_train, classDict['truck']),
        get_class_i(x_train, y_train, classDict['ship'])
    ],
    transform_with_aug
)


testsetLoader = DataLoader(test_avian_land_dataset,
                           batch_size=5000,
                           shuffle=True,
                           num_workers=8,
                           pin_memory=True)

trainsetLoader = DataLoader(avian_land_dataset,
                            batch_size=5000,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True)


def train_multiclass_model(model, train_loader, val_loader, loss, optimizer, num_epoch):
    print('Training...')

    # Initialize history
    loss_history = []
    val_loss_history = []
    accuracy_history = []
    val_accuracy_history = []
    test_accuracy_history = []

    for epoch in tqdm(range(num_epoch)):
        tqdm.write('Epoch: {}/{}'.format(epoch+1, num_epoch))
        model.train()
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        correct_samples_total = 0
        total_samples_total = 0

        for i_step, (x, y_class, one_hot_vectors) in enumerate(train_loader):

            indices_one = [i for i, x in enumerate(y_class) if
                           (x == 'plane') | (x == 'cat')]
            indices_two = [i for i, x in enumerate(y_class) if
                           (x == 'frog') | (x == 'ship')]

            x = x.to(device)
            one_hot_vectors = one_hot_vectors.to(device)
            prediction = model(x)

            loss_value = loss(
                prediction[indices_one][:, :4], one_hot_vectors[indices_one][:, :4])

            loss_value += loss(
                prediction[indices_two], one_hot_vectors[indices_two])

            _, indices_pred = torch.max(prediction[:, :4], 1)
            _, indices_true = torch.max(one_hot_vectors[:, :4], 1)

            correct_samples += torch.sum(indices_pred == indices_true)
            total_samples += indices_pred.shape[0]

            total_samples_total += indices_pred.shape[0]

            # Optimize model
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_accum += loss_value

            # writer.add_graph(model, x_two)

        ave_loss = loss_accum/i_step

        train_accuracy = float(correct_samples)/total_samples
        test_accuracy, val_accuracy, val_loss = calculate_accuracy(
            model, val_loader)

        val_accuracy_history.append(val_accuracy)
        test_accuracy_history.append(test_accuracy)
        accuracy_history.append(train_accuracy)
        val_loss_history.append(float(val_loss))
        loss_history.append(float(ave_loss))

        writer.add_scalar('Loss/train', ave_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        tqdm.write('Loss_train: {}\tLoss_val:{}'.format(ave_loss, val_loss))
        tqdm.write('Acc_train: {}\t\tAcc_val: {}\tAcc_test: {}'.format(
            train_accuracy, val_accuracy, test_accuracy))
        tqdm.write('')

    print('Training finished')
    return loss_history, accuracy_history, val_loss_history, val_accuracy_history, test_accuracy_history


def calculate_accuracy(model, loader):
    model.eval()
    correct_associations = 0
    correct_distinctions = 0
    total_samples = 0
    loss_accum = 0
    with torch.no_grad():
        for i_step, (x, y_class, y_oh) in enumerate(loader):
            test_indices = [i for i, x in enumerate(y_class) if
                            (x == 'plane') | (x == 'cat')]

            x = x.to(device)[test_indices]
            y_association = y_oh.to(device)[test_indices][:, :2]

            y_validation = y_oh.to(device)[test_indices]

            prediction = model(x)
            association = prediction[:, 4:]
            tqdm.write(
                f"{y_validation[0]}, {y_association[0]}, {prediction[0]}")

            distinction = prediction[:, :4]

            loss_accum += loss(association, y_association)
            _, indices_association = torch.max(association, 1)
            _, true_associations = torch.max(y_association, 1)
            _, indices_distinctions = torch.max(distinction, 1)
            _, true_distinctions = torch.max(y_validation, 1)

           
            correct_associations += torch.sum(
                indices_association == true_associations
            )

            correct_distinctions += torch.sum(
                indices_distinctions == true_distinctions)
            total_samples += indices_association.shape[0]

    accuracy_association = float(correct_associations)/total_samples
    accuracy_distinction = float(correct_distinctions)/total_samples
    ave_loss = loss_accum/i_step if i_step > 0 else loss_accum
    return accuracy_association, accuracy_distinction, ave_loss


model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = (nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 6),
))
# print(model)

model = model.to(device)
loss = nn.BCEWithLogitsLoss()
# loss = nn.BCELoss()
optimizer = optim.Adam(model.parameters())  

loss_history,              \
    accuracy_history,      \
    val_loss_history,      \
    val_accuracy_history,  \
    test_accuracy_history = train_multiclass_model(model,
                                                   trainsetLoader,
                                                   testsetLoader,
                                                   loss,
                                                   optimizer,
                                                   50)

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(accuracy_history, label='Train')
plt.plot(val_accuracy_history, label='Validation')
plt.plot(test_accuracy_history, label='Test')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss_history, label='Train')
plt.plot(val_loss_history, label='Validation')
plt.title('Loss')
plt.legend()

plt.savefig(f'figures/{run_name}.png')

print(max(test_accuracy_history), min(val_loss_history))
