#!/usr/bin/env python
# coding: utf-8

# Importing libraries
import warnings
from sklearn import preprocessing
from skimage.io import imread
from skimage import transform
import numpy as np
import pandas as pd
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


run_name = "resnet18_not_pretrained_4_classes"

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

# Separating trainset/testset data/label
x_train = trainset.data
x_test = testset.data
y_train = trainset.targets
y_test = testset.targets


class_label_mapper = {
    0: [0, 2],
    1: [1, 3],
    2: [0, 2],
    3: [1, 3]
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
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)

        labels = [class_label_mapper[class_label]]

        return img, trainset.classes[class_label], self.to_onehot(labels, 4)

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
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

    def to_onehot(self, labels, n_categories, dtype=torch.float32):
        one_hot_labels = torch.zeros(size=(n_categories, ), dtype=dtype)
        for label in labels:
            one_hot_labels[label] = 1.
        return one_hot_labels



avian_land_dataset = DatasetMaker(
    [get_class_i(x_train, y_train, classDict['plane']),
     get_class_i(x_train, y_train, classDict['car']),
     get_class_i(x_train, y_train, classDict['bird']),
     get_class_i(x_train, y_train, classDict['cat'])],
    transform_with_aug
)


test_avian_land_dataset = DatasetMaker(
    [get_class_i(x_test, y_test, classDict['plane']),
     get_class_i(x_test, y_test, classDict['car']),
     get_class_i(x_test, y_test, classDict['bird']),
     get_class_i(x_test, y_test, classDict['cat'])],
    transform_with_aug
)


testsetLoader = DataLoader(test_avian_land_dataset,
                           batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
trainsetLoader = DataLoader(
    avian_land_dataset, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)


def train_multiclass_model(model, train_loader, val_loader, loss, optimizer, num_epoch):
    print('Training...')

    # Initialize history
    loss_history = []
    val_loss_history = []
    accuracy_history = []
    val_accuracy_history = []

    for epoch in tqdm(range(num_epoch)):
        print('Epoch: {}/{}'.format(epoch+1, num_epoch))
        model.train()
        loss_accum = 0
        correct_samples = 0
        total_samples = 0
        correct_samples_total = 0
        total_samples_total = 0

        for i_step, (x, y_class, one_hot_vectors) in enumerate(train_loader):
            #             y_oh=y_oh.type(torch.FloatTensor)
            # Divide classes into pasta/hotdog (without ticker of country) and burger/pizza (with ticker of country)
            
            indexes_one = [i for i, x in enumerate(y_class) if (
                x == 'airplane') | (x == 'automobile')]
            indexes_two = [i for i, x in enumerate(
                y_class) if (x == 'bird') | (x == 'cat')]

            # Calculate loss for 1st part (2 one-hot classes, pasta and hotdog) (classes 3,4 are cutted off)
            x_one, y_oh_one = x[indexes_one], one_hot_vectors[indexes_one][:, :2]
            x_one, y_oh_one = x_one.to(device), y_oh_one.to(device)
            prediction_one = model(x_one)

            loss_value_one = loss(prediction_one[:, :2], y_oh_one)

            # Calculate loss for 2nd part (4 one-hot classes, pizza and burger)
            x_two, y_oh_two = x[indexes_two], one_hot_vectors[indexes_two]
            x_two, y_oh_two = x_two.to(device), y_oh_two.to(device)

            y_oh_two = y_oh_two[:, :4]
            prediction_two = model(x_two)
            loss_value_two = loss(prediction_two, y_oh_two)

            # Sum up losses
            loss_value = loss_value_one+loss_value_two

            # Calculate accuracy of classes 3/4 for pasta/hotdog, which is not learned strictly by net,
            # but by association through pizza & burger
            # took only predictions of pasta/hotdog for classes 3/4
            prediction = prediction_one[:, 2:]
            _, indices_pred = torch.max(prediction, 1)
            # one-hot encoding for 3/4 looks same as for 1/2
            _, indices_true = torch.max(y_oh_one, 1)
            correct_samples += torch.sum(indices_pred == indices_true)
            total_samples += indices_pred.shape[0]

            # Calculate accuracy of calsses 1/2 for all images in a batch
            x_total, y_total = x, one_hot_vectors[:, :2]
            x_total, y_total = x_total.to(device), y_total.to(device)

            prediction_total = model(x_total)[:, :2]
            _, indices_total_pred = torch.max(prediction_total, 1)
            _, indices_total_true = torch.max(y_total, 1)
            correct_samples_total += torch.sum(
                indices_total_pred == indices_total_true)
            total_samples_total += indices_total_pred.shape[0]

            # Optimize model
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_accum += loss_value

            # writer.add_graph(model, x_two)


        ave_loss = loss_accum/i_step

        train_accuracy = float(correct_samples)/total_samples
        class_accuracy_train = float(correct_samples_total)/total_samples_total
        val_accuracy, val_loss = calculate_accuracy(model, val_loader)

        val_accuracy_history.append(val_accuracy)
        accuracy_history.append(train_accuracy)
        val_loss_history.append(float(val_loss))
        loss_history.append(float(ave_loss))

        writer.add_scalar('Loss/train', ave_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', val_accuracy, epoch)
        
        print('Loss_train: {}, Loss_val:{}'.format(ave_loss, val_loss))
        print('Acc_train: {}, Acc_val: {}'.format(train_accuracy, val_accuracy))
        print('Acc_classifier: {}'.format(class_accuracy_train))

    print('Learning is finished')
    return loss_history, accuracy_history, val_loss_history, val_accuracy_history


def calculate_accuracy(model, loader):
    model.eval()
    correct_samples = 0
    total_samples = 0
    loss_accum = 0
    with torch.no_grad():
        for i_step, (x, y_class, y_oh) in enumerate(loader):
            #             print(y_class)
            indexes_one = [i for i, x in enumerate(
                y_class) if (x == 'airplane') | (x == 'car')]
            y_oh = y_oh.type(torch.FloatTensor)
            y_class = list(y_class)
            x, y = x.to(device)[indexes_one], y_oh.to(
                device)[indexes_one][:, :2]
            prediction = model(x)[:, 2:]
            loss_accum += loss(prediction, y)
            _, indices_pred = torch.max(prediction, 1)
            _, indices_true = torch.max(y, 1)
            correct_samples += torch.sum(indices_pred == indices_true)
            total_samples += indices_pred.shape[0]
    accuracy = float(correct_samples)/total_samples
    ave_loss = loss_accum/i_step
    print(i_step)
    return accuracy, ave_loss


# My model

multiclass_net = models.resnet18(pretrained=False)
num_ftrs = multiclass_net.fc.in_features
multiclass_net.fc = (nn.Linear(num_ftrs, 4))

# multiclass_net.load_state_dict(torch.load('classifier.pt'))
multiclass_net = multiclass_net.to(device)
loss = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(multiclass_net.parameters(), lr=1e-3)  # , momentum=0.9)

loss_history, accuracy_history, val_loss_history, val_accuracy_history = train_multiclass_model(
    multiclass_net, trainsetLoader, testsetLoader, loss, optimizer, 50)
#torch.save(multiclass_net.state_dict(), 'multiclass_v2.pt')

plt.figure(figsize=(20, 5))
plt.subplot(1, 2, 1)
plt.plot(accuracy_history, label='Train')
plt.plot(val_accuracy_history, label='Validation')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(loss_history, label='Train')
plt.plot(val_loss_history, label='Validation')
plt.title('Loss')
plt.legend()

plt.savefig(f'figures/{run_name}.png')

print(max(val_accuracy_history), min(val_loss_history))
