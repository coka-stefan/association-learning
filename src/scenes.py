from lxml import etree as ET
import torchvision.datasets as dset

import argparse

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
import torch.nn.functional as F

import torchvision

from torchvision import transforms, models
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
from PIL import ImageFile, ImageDraw

from statistics import mean

from sklearn.metrics import f1_score, precision_score

ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch_size", type=int, default=64,
                    help="batch size")
parser.add_argument("-e", "--num_epochs", type=int, default=1,
                    help="maximum number of epochs to train")
parser.add_argument("-s", "--shuffle_dataset", type=bool, default=True,
                    help="shuffle train dataset?")
parser.add_argument("-v", "--val_split", type=float, default=0.2,
                    help="validation split portion")
parser.add_argument("-r", "--run_name", help="name of the experiment")

parser.add_argument("-c", "--crop", type=bool, default=False,
                    help="should testing be done on crops?")

args = parser.parse_args()

batch_size = args.batch_size
shuffle_dataset = args.shuffle_dataset
validation_split = args.val_split
run_name = args.run_name
num_epochs = args.num_epochs
should_crop = args.crop

print(f'Should crop? {should_crop}')

writer = SummaryWriter(f'runs/{run_name}')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")


RS = transforms.Resize((256))
RC = transforms.RandomCrop(256, pad_if_needed=True)
RHF = transforms.RandomHorizontalFlip()
RVF = transforms.RandomVerticalFlip()
RR = transforms.RandomRotation((-10, 10))
NRM = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TT = transforms.ToTensor()
TPIL = transforms.ToPILImage(mode='RGB')

inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)

# Transforms object for trainset with augmentation
transform_with_aug = transforms.Compose([RS, RR, RHF, TT, NRM])
# Transforms object for testset with NO augmentation
transform_no_aug = transforms.Compose([RS, TT, NRM])
transform_no_aug_crop = transforms.Compose([RC, TT, NRM])


objects = {
    'kitchen': [
        # 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        # 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        # 'hot dog', 'pizza', 'donut', 'cake', 'diningtable',
        'microwave',
        'oven',
        'toaster',
        'sink',
        'refrigerator'
    ],

    'living_room': [
        'chair',
        'sofa',
        'pottedplant',
        'tvmonitor',
    ]
}


def read_content(xml_file: str):
    '''
    Read bounding boxes from xml file
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):
        ymin, xmin, ymax, xmax = None, None, None, None
        name = boxes.find('name').text

        for box in boxes.findall("bndbox"):
            ymin = float(box.find("ymin").text)
            xmin = float(box.find("xmin").text)
            ymax = float(box.find("ymax").text)
            xmax = float(box.find("xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append((list_with_single_boxes, name))

    return list_with_all_boxes


class ImageFolderWithNames(dset.ImageFolder):

    def __getitem__(self, i):
        return super(ImageFolderWithNames, self).__getitem__(i), self.imgs[i]


def getDataLoaders():

    dataset = ImageFolderWithNames(
        '../data/', transform=transform_no_aug)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              num_workers=8,
                              sampler=train_sampler)

    validation_loader = DataLoader(dataset,
                                   batch_size=batch_size,
                                   num_workers=0,
                                   sampler=val_sampler)

    return train_loader, validation_loader, dataset.classes


def trainModel(model, train_loader, val_loader, loss, optimizer, num_epoch, classes, test_bboxes):
    print('Training...')

    # Initialize history
    loss_history = []
    val_loss_history = []
    accuracy_history = []
    val_accuracy_history = []
    test_accuracy_history = []
    f1_history = []

    for epoch in tqdm(range(num_epoch), desc='Epochs', dynamic_ncols=True):
        tqdm.write('Epoch: {}/{}'.format(epoch+1, num_epoch))
        model.train()
        loss_accumulated = 0
        correct_samples = 0
        total_samples = 0
        total_samples_total = 0

        for i_step, load in tqdm(enumerate(train_loader),
                                 leave=False,
                                 total=len(train_loader),
                                 desc='Batches',
                                 dynamic_ncols=True):

            img, _ = load

            x, y = img

            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            
            _, indices_pred = torch.max(prediction, 1)
            correct_samples += torch.sum(indices_pred == y)

            loss_value = loss(prediction, y)

            total_samples += indices_pred.shape[0]
            total_samples_total += indices_pred.shape[0]

            # Optimize model
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_accumulated += loss_value

        ave_loss = loss_accumulated/i_step

        train_accuracy = float(correct_samples)/total_samples
        val_accuracy, val_loss, test_accuracy, test_loss, f1_test = calculate_accuracy(
            model,
            val_loader,
            epoch,
            classes,
            test_bboxes)

        val_accuracy_history.append(val_accuracy)
        test_accuracy_history.append(test_accuracy)
        accuracy_history.append(train_accuracy)
        val_loss_history.append(float(val_loss))
        loss_history.append(float(ave_loss))
        f1_history.append(f1_test)

        writer.add_scalar('Loss/train', ave_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Loss/test', test_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/test', test_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        tqdm.write(
            f'Loss_train: {ave_loss}\tLoss_val:{val_loss} \tLoss_test:{test_loss}'
        )
        tqdm.write(
            f'Acc_train: {train_accuracy}\t\tAcc_val: {val_accuracy}\tAcc_test: {test_accuracy}'
        )
        tqdm.write(
            f'F1_test: {f1_test}'
        )
        tqdm.write('')

    return loss_history, accuracy_history, val_loss_history, val_accuracy_history, test_accuracy_history, f1_history


def get_bboxes(loader):
    bboxes = {}
    for i_step, load in tqdm(loader, desc='Bounding boxes'):
        img_names, _ = load
        for i in img_names:
            img_location = "/".join(i.split('/')[-2:]).split('.')[0]
            bboxes[i] = read_content(
                f"../places/data_256/bboxes/{img_location}.xml")
    return bboxes


def calculate_accuracy(model, loader, train_step, classes, test_bboxes):
    model.eval()
    correct_samples = 0
    correct_samples_manip = 0
    total_samples = 0
    loss_val = []
    loss_test = []
    f1_test = []
    total_manip_samples = 0

    ground_truths_manip = []
    all_predictions_manip = []

    elephant = Image.open('./random_object.jpg')
    elephant = elephant.resize((256, 256))
    elephant = TT(elephant)
    elephant = NRM(elephant)
    elephant.unsqueeze_(0)

    with torch.no_grad():
        t = tqdm(enumerate(loader), leave=False,
                 total=len(loader), desc='Validation', dynamic_ncols=True)
        for i_step, load in t:
            imgs, img_names = load
            # bboxes = []
            # for i in img_names[0]:
            #     img_location = "/".join(i.split('/')[-2:]).split('.')[0]
            #     bboxes.append(read_content(
            #         f"../places/data_256/bboxes/{img_location}.xml"))

            bboxes = []
            for i in img_names[0]:
                bboxes.append(test_bboxes[i])

            x, y = imgs

            x_manip = []
            y_manip = []

            for i, (img, img_bbox) in enumerate(zip(x, bboxes)):

                if not should_crop:
                    if img_bbox:

                        total_manip_samples += 1
                        y_manip.append(y[i])
                        for bbox, name in img_bbox:
                            if name in objects[classes[y[i].item()]]:
                                x0, y0, x1, y1 = [
                                    int(b) if b >= 0 else 0 for b in bbox]
                                if x0 != x1 and y0 != y1:
                                    _, w, h = img[:, y0:y1, x0:x1].shape

                                    replace = F.interpolate(elephant,
                                                            size=(w, h),
                                                            mode='bicubic',
                                                            align_corners=True)

                                    img[:, y0:y1, x0:x1] = replace

                        x_manip.append(img.unsqueeze_(0))

                elif should_crop:
                    if img_bbox:
                        for bbox, name in img_bbox:
                            if name in objects[classes[y[i].item()]]:
                                x0, y0, x1, y1 = [
                                    int(b) if b >= 0 else 0 for b in bbox]
                                if x0 != x1 and y0 != y1:
                                    total_manip_samples += 1
                                    y_manip.append(y[i])

                                    im = img[:, y0:y1, x0:x1]
                                    target = F.interpolate(im.unsqueeze_(0),
                                                           size=256,
                                                           mode='bicubic',
                                                           align_corners=False)
                                    x_manip.append(target)

            x = x.to(device)
            y = y.to(device)

            x_manip = torch.cat(x_manip)
            x_manip = x_manip.to(device)
            y_manip = torch.tensor(y_manip).to(device)

            if train_step == 0:
                writer.add_histogram(
                    'Distribution of test labels', y_manip, i_step)

            predictions = model(x)
            predictions_manip = model(x_manip)

            # # Uncomment if grid of testing images needs to be written to tensorboard
            # if train_step == 0:
            #     ims = []
            #     for im in x_manip:
            #         ims.append(inv_normalize(im).unsqueeze_(0))
            #     ims = torch.cat(ims)
            #     grid = torchvision.utils.make_grid(ims)
            #     writer.add_image('test_images', grid, i_step)

            # # Uncomment if test predictions need to be written to tensorboard
            # for pred, im, true in zip(predictions_manip, x_manip, y_manip):
            #     _, class_ = pred.max(0)
            #     im = inv_normalize(im)
            #     # print(classes[int(class_)], classes[true])
            #     writer.add_image(
            #         f'{classes[int(class_)]} | {classes[int(true)]}',
            #         im,
            #         i_step)

            loss_val.append(loss(predictions, y))
            loss_test.append(loss(predictions_manip, y_manip))

            _, indices_pred = torch.max(predictions, 1)
            _, indices_pred_manip = torch.max(predictions_manip, 1)
            # _, indices_true = torch.max(y, 1)
            correct_samples += torch.sum(indices_pred == y)
            correct_samples_manip += torch.sum(indices_pred_manip == y_manip)

            total_samples += indices_pred.shape[0]

            accuracy = float(correct_samples)/total_samples
            accuracy_manip = float(correct_samples_manip)/total_manip_samples

            ground_truths_manip.append(y_manip)
            all_predictions_manip.append(indices_pred_manip)

            t.set_postfix(
                accuracy=accuracy,
                test_accuracy=accuracy_manip)

    tqdm.write(f'total number of test samples: {total_manip_samples}')
    tqdm.write(f'total number of val samples: {total_samples}')

    ave_loss = torch.mean(torch.tensor(loss_val))
    ave_loss_test = torch.mean(torch.tensor(loss_test))

    ground_truths_manip = torch.cat(ground_truths_manip)
    all_predictions_manip = torch.cat(all_predictions_manip)
    f1_test = f1_score(ground_truths_manip.cpu(), all_predictions_manip.cpu())

    return accuracy, ave_loss, accuracy_manip, ave_loss_test, f1_test


def buildModel():
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = (nn.Sequential(
        nn.Linear(num_features, 2),
    ))

    model = nn.DataParallel(model)

    # print(model)
    model = model.to(device)

    return model


train_loader, validation_loader, classes = getDataLoaders()
model = buildModel()
loss = nn.CrossEntropyLoss()
# loss = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=10e-4)

test_bboxes = get_bboxes(validation_loader)

loss_history,              \
    accuracy_history,      \
    val_loss_history,      \
    val_accuracy_history,  \
    test_accuracy_history, \
    f1_score_history = trainModel(model,
                                  train_loader,
                                  validation_loader,
                                  loss,
                                  optimizer,
                                  num_epochs,
                                  classes,
                                  test_bboxes)

plt.figure(figsize=(20, 5))
plt.subplot(1, 3, 1)
plt.plot(accuracy_history, label='Train')
plt.plot(val_accuracy_history, label='Validation')
plt.plot(test_accuracy_history, label='Test')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(loss_history, label='Train')
plt.plot(val_loss_history, label='Validation')
plt.title('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(f1_score_history, label='Test')
plt.title('F1 Score')
plt.legend()

plt.savefig(f'figures/{run_name}.png')

print(max(test_accuracy_history), min(val_loss_history))
