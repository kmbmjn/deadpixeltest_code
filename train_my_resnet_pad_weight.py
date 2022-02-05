from __future__ import print_function, division
import time
import copy
import os
import argparse
from termcolor import colored
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

# from torchvision import models, transforms
# from efficientnet_pytorch import EfficientNet
# from efficientnet_pytorch.utils import Conv2dStaticSamePadding

import cv2
import pdb
from my_model import *
import random
from lmfit import Model

np.random.seed(0)

parser = argparse.ArgumentParser(description="resnet_teacher")
parser.add_argument("--model_name", default="my_resnet18_pad", type=str)
parser.add_argument("--mode", default="train", type=str)
parser.add_argument("--data_name", default="cal", type=str)
parser.add_argument("--hp_lr", type=float, default=1e-2)
parser.add_argument("--hp_wd", type=float, default=5e-4)
parser.add_argument("--hp_bs", type=int, default=128)
parser.add_argument("--hp_ep", type=int, default=200)
parser.add_argument("--hp_opt", type=str, default="sgd")
parser.add_argument("--hp_sch", type=str, default="cos")
parser.add_argument("--hp_ax", type=int, default=100)
parser.add_argument("--hp_ay", type=int, default=100)
parser.add_argument("--hp_tr", type=int, default=1)
parser.add_argument("--hp_le", type=int, default=1)
parser.add_argument("--hp_lx", type=int, default=100)
parser.add_argument("--hp_ly", type=int, default=100)
parser.add_argument("--hp_nc", type=int, default=2)
parser.add_argument("--hp_rs", type=int, default=1)
parser.add_argument("--hp_mn", type=int, default=14)
parser.add_argument("--hp_nw", type=int, default=1)
parser.add_argument("--hp_id", type=int, default=0)
parser.add_argument("--hp_no", type=float, default=1e-4)

opt = parser.parse_args()
model_name = opt.model_name
hp_lr = opt.hp_lr
hp_wd = opt.hp_wd
hp_bs = opt.hp_bs
hp_ep = opt.hp_ep
hp_opt = opt.hp_opt
hp_sch = opt.hp_sch
hp_tr = opt.hp_tr
hp_le = opt.hp_le
hp_lx = opt.hp_lx
hp_ly = opt.hp_ly
hp_nc = opt.hp_nc
hp_rs = opt.hp_rs
hp_mn = opt.hp_mn
data_name = opt.data_name

# hp_ep = 2


if opt.mode == "grad":
    hp_bs = 1

if opt.mode == "grad_dAdi":
    hp_bs = 1
if opt.mode == "grad_dAdi_layer3":
    hp_bs = 1

if opt.mode == "attack":
    hp_bs = 1
    torch.manual_seed(0)

if opt.mode == "conf":
    hp_bs = 1

if opt.mode == "noise_confdrop":
    hp_bs = 1


transform_aug = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform = transforms.Compose(
    [
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_trans = transforms.Compose(
    [
        transforms.Resize(size=256),
        # transforms.CenterCrop(size=224),
        # transforms.CenterCrop(size=225),
        transforms.CenterCrop(size=224+hp_tr),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

transform_micro_mnist = transforms.Compose(
    [
        transforms.Resize(size=opt.hp_mn),
        transforms.ToTensor(),
    ]
)

if data_name == "cal":
    dataset_aug = ImageFolder(root="../s_dataset/101_ObjectCategories/", transform=transform_aug)
    dataset = ImageFolder(root="../s_dataset/101_ObjectCategories/", transform=transform)
    dataset_trans = ImageFolder(root="../s_dataset/101_ObjectCategories/", transform=transform_trans)
    num_classes = 102
elif data_name == "cub":
    dataset_aug = ImageFolder(root="../s_dataset/cub/CUB_200_2011/images/", transform=transform_aug)
    dataset = ImageFolder(root="../s_dataset/cub/CUB_200_2011/images/", transform=transform)
    dataset_trans = ImageFolder(root="../s_dataset/cub/CUB_200_2011/images/", transform=transform_trans)
    num_classes = 200
elif data_name == "mnist":
    train_dataset_mnist = torchvision.datasets.MNIST(root="./MNIST_Dataset", transform=transform_micro_mnist, train=True, download=True)
    test_dataset_mnist = torchvision.datasets.MNIST(root="./MNIST_Dataset", transform=transform_micro_mnist, train=False, download=True)
elif data_name == "pet":
    dataset_aug = ImageFolder(root="~/dataset/oxford_pet/images/", transform=transform_aug)
    dataset = ImageFolder(root="~/dataset/oxford_pet/images/", transform=transform)
    dataset_trans = ImageFolder(root="~/dataset/oxford_pet/images/", transform=transform_trans)
    num_classes = 37


if opt.mode == "train_micro_object":
    num_classes = 2
elif opt.mode == "train_micro_object_rgb":
    num_classes = hp_nc
elif opt.mode == "train_micro_object_rgb_rgb":
    num_classes = hp_nc
elif opt.mode == "translate_micro":
    num_classes = 2
elif opt.mode == "train_micro_mnist":
    num_classes = 10

if opt.mode not in ["train_micro_mnist"]:
    # Shuffle the indices
    len_dataset = len(dataset)  # 9144
    len_train = int(len_dataset * 0.7)  # 6400
    len_val = int(len_dataset * 0.15)  # 1371
    len_test = int(len_dataset * 0.15)  # 1374

    indices = np.arange(0, len_dataset)
    np.random.shuffle(indices)  # shuffle the indicies

        # https://medium.com/jun-devpblog/pytorch-5-pytorch-visualization-splitting-dataset-save-and-load-a-model-501e0a664a67

    train_loader = torch.utils.data.DataLoader(
        # dataset,
        dataset_aug,
        batch_size=hp_bs,
        shuffle=False,
        num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:len_train]),
    )

    val_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hp_bs,
        shuffle=False,
        num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(
            indices[len_train : len_train + len_val]
        ),
    )

    val_loader_trans = torch.utils.data.DataLoader(
        # dataset,
        dataset_trans,
        batch_size=hp_bs,
        shuffle=False,
        num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(
            indices[len_train : len_train + len_val]
        ),
    )

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=hp_bs,
        shuffle=False,
        num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(
            indices[len_train + len_val : len_train + len_val + len_test]
        ),
    )

    test_loader_trans = torch.utils.data.DataLoader(
        # dataset,
        dataset_trans,
        batch_size=hp_bs,
        shuffle=False,
        num_workers=0,
        sampler=torch.utils.data.SubsetRandomSampler(
            indices[len_train + len_val : len_train + len_val + len_test]
        ),
    )

if opt.mode == "train_micro_mnist":
    len_dataset = len(train_dataset_mnist)  # 60000
    len_train = 50000
    len_val = 10000
    len_test = 10000

    indices = np.arange(0, len_dataset)
    np.random.shuffle(indices)  # shuffle the indicies

    train_loader = torch.utils.data.DataLoader(
        train_dataset_mnist,
        batch_size=hp_bs,
        shuffle=False,
        num_workers=opt.hp_nw,
        sampler=torch.utils.data.SubsetRandomSampler(indices[:len_train]),
        pin_memory = True,
    )

    val_loader = torch.utils.data.DataLoader(
        train_dataset_mnist,
        batch_size=hp_bs,
        shuffle=False,
        num_workers=opt.hp_nw,
        sampler=torch.utils.data.SubsetRandomSampler(
            indices[len_train : len_train + len_val]
        ),
        pin_memory = True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset_mnist,
        batch_size=hp_bs,
        shuffle=False,
        num_workers=opt.hp_nw,
        pin_memory = True,
        # sampler=torch.utils.data.SubsetRandomSampler(
        #     indices[len_train + len_val : len_train + len_val + len_test]
        # ),
    )


dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}

loader_sizes = {
    "train": len(train_loader),
    "val": len(val_loader),
    "test": len(test_loader),
}

dataset_sizes = {"train": len_train, "val": len_val, "test": len_test}

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:" + str(opt.hp_id) if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")  # 가장 작은 수

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase

        for phase in ["train", "val", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{}_Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                print(colored("It is now best.", "green"))
                best_acc = epoch_acc
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best epoch: ", best_epoch)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # return model
    return [model, best_acc.cpu().numpy(), best_loss]


def put_micro_object(input_single):

    len_object = hp_le

    loc_x = random.randrange(0, 225-len_object)  # 0, 1, ..., 223
    loc_y = random.randrange(0, 225-len_object)

    input_single[0, loc_x:loc_x+len_object, loc_y:loc_y+len_object] = (1.-0.485)/0.2290
    input_single[1, loc_x:loc_x+len_object, loc_y:loc_y+len_object] = (0.-0.456)/0.224
    input_single[2, loc_x:loc_x+len_object, loc_y:loc_y+len_object] = (0.-0.406)/1.225

    return input_single


def put_micro_object_at(input_single, loc_x, loc_y):

    input_single[0, loc_x, loc_y] = (1.-0.485)/0.2290
    input_single[1, loc_x, loc_y] = (0.-0.456)/0.224
    input_single[2, loc_x, loc_y] = (0.-0.406)/1.225

    return input_single



def train_model_micro_object(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")  # 가장 작은 수

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase

        for phase in ["train", "val", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # inputs [128, 3, 224, 224], float32
                # labels [128], int64
                # labels는 4, 95같은 정수 들임.
                len_labels = labels.size()
                inputs_micro = inputs.clone().detach()
                inputs_micro = torch.stack([put_micro_object(input_single) for input_single in inputs_micro])  # 123, 3, 224, 224
                inputs = torch.cat([inputs, inputs_micro], dim=0)  # 256, 3, 224, 224, float32
                labels = torch.cat([torch.zeros(len_labels), torch.ones(len_labels)])  # 256
                labels = labels.type(torch.int64)
                #

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / (dataset_sizes[phase] * 2)
            epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 2)

            print("{}_Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                print(colored("It is now best.", "green"))
                best_acc = epoch_acc
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best epoch: ", best_epoch)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # return model
    return [model, best_acc.cpu().numpy(), best_loss]


def put_micro_object_rgbint(input_single, rint, gint, bint):
    input_single = input_single.clone().detach()

    rfloat = rint / 255.
    gfloat = gint / 255.
    bfloat = bint / 255.

    len_object = hp_le

    loc_x = random.randrange(0, 225-len_object)  # 0, 1, ..., 223
    loc_y = random.randrange(0, 225-len_object)

    input_single[0, loc_x:loc_x+len_object, loc_y:loc_y+len_object] = (rfloat-0.485)/0.2290
    input_single[1, loc_x:loc_x+len_object, loc_y:loc_y+len_object] = (gfloat-0.456)/0.224
    input_single[2, loc_x:loc_x+len_object, loc_y:loc_y+len_object] = (bfloat-0.406)/1.225

    return input_single


def train_model_micro_object_rgbint(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")  # 가장 작은 수

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase

        for phase in ["train", "val", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # inputs [128, 3, 224, 224], float32
                # labels [128], int64
                # labels는 4, 95같은 정수 들임.

                ## inputs = torch.cat([torch.stack([put_micro_object_rgbint(input_single, rint, 0, 0) for rint in range(0, 256, r_step)]) for input_single in inputs], dim=0)  # 32, 3, 224, 224
                inputs = torch.cat([torch.stack([put_micro_object_rgbint(input_single, rint, 0, 0) for rint in range(0, num_classes*hp_rs, hp_rs)]) for input_single in inputs], dim=0)  # 32, 3, 224, 224

                len_labels = labels.size()[0]
                labels = torch.cat([torch.arange(0, num_classes, 1, dtype=torch.int64) for i in range(len_labels)])  # 48

                labels = labels.type(torch.int64)
                # tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                #                 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # epoch_loss = running_loss / (dataset_sizes[phase] * 2)
            # epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 2)
            ## epoch_loss = running_loss / (dataset_sizes[phase] * 26)
            ## epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 26)
            epoch_loss = running_loss / (dataset_sizes[phase] * num_classes)
            epoch_acc = running_corrects.double() / (dataset_sizes[phase] * num_classes)

            print("{}_Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                print(colored("It is now best.", "green"))
                best_acc = epoch_acc
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best epoch: ", best_epoch)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # return model
    return [model, best_acc.cpu().numpy(), best_loss]


def train_model_micro_object_rgbint_rgb(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")  # 가장 작은 수

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase

        for phase in ["train", "val", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # inputs [128, 3, 224, 224], float32
                # labels [128], int64
                # labels는 4, 95같은 정수 들임.

                ## inputs = torch.cat([torch.stack([put_micro_object_rgbint(input_single, rint, 0, 0) for rint in range(0, 256, r_step)]) for input_single in inputs], dim=0)  # 32, 3, 224, 224
                ## inputs = torch.cat([torch.stack([put_micro_object_rgbint(input_single, rint, 0, 0) for rint in range(0, num_classes*hp_rs, hp_rs)]) for input_single in inputs], dim=0)  # 32, 3, 224, 224
                inputs = torch.cat([torch.stack([put_micro_object_rgbint(input_single, rint, rint, rint) for rint in range(0, num_classes*hp_rs, hp_rs)]) for input_single in inputs], dim=0)  # 32, 3, 224, 224

                len_labels = labels.size()[0]
                labels = torch.cat([torch.arange(0, num_classes, 1, dtype=torch.int64) for i in range(len_labels)])  # 48

                labels = labels.type(torch.int64)
                # tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2,
                #                 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            # epoch_loss = running_loss / (dataset_sizes[phase] * 2)
            # epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 2)
            ## epoch_loss = running_loss / (dataset_sizes[phase] * 26)
            ## epoch_acc = running_corrects.double() / (dataset_sizes[phase] * 26)
            epoch_loss = running_loss / (dataset_sizes[phase] * num_classes)
            epoch_acc = running_corrects.double() / (dataset_sizes[phase] * num_classes)

            print("{}_Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                print(colored("It is now best.", "green"))
                best_acc = epoch_acc
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best epoch: ", best_epoch)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # return model
    return [model, best_acc.cpu().numpy(), best_loss]


def pad_micro_mnist(input_single):
    # input_single.size(): 1, 14, 14
    # input_single = input_single.clone().detach().view(1, 1, 14, 14)  # clone 때문에 다른 메모리의 tensor임
    input_single = torch.unsqueeze(input_single.clone().detach(), dim=0)  # clone 때문에 다른 메모리의 tensor임
    max_length_x = 224 - opt.hp_mn
    max_length_y = 224 - opt.hp_mn
    pad_x = random.randrange(0, max_length_x)  # 0, 1, ..., 209
    pad_y = random.randrange(0, max_length_y)  # 0, 1, ..., 209
    pad_dim = (pad_y, max_length_y-pad_y, pad_x, max_length_x-pad_x, 1, 1, 0, 0)  # 축이 끝부분 부터 먼저임. 1 1로 gray를 RGB로.
    # input_single = F.pad(input_single, pad_dim, "constant", 0).view(1, 3, 224, 224)
    # input_single = torch.unsqueeze(F.pad(input_single, pad_dim, "constant", 0), dim=0)
    input_single = F.pad(input_single, pad_dim, "constant", 0)
    # input_single: 1, 3, 224, 224

    return input_single


def train_model_micro_mnist(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = float("-inf")  # 가장 작은 수

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Each epoch has a training and validation phase

        for phase in ["train", "val", "test"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                # inputs: 32, 1, 14, 14
                # inputs[0]: 1, 14, 14

                inputs = torch.cat([pad_micro_mnist(input_single) for input_single in inputs], dim=0)  # checked -> 32, 3, 224, 224

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print("{}_Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val" and epoch_acc > best_acc:
                print(colored("It is now best.", "green"))
                best_acc = epoch_acc
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    print("Best epoch: ", best_epoch)

    # load best model weights
    model.load_state_dict(best_model_wts)

    # return model
    return [model, best_acc.cpu().numpy(), best_loss]


save_dir = "retrain_model_folder/pt_" + str(model_name) + "_" + str(hp_lr)

model_ft, model_type = get_my_model(model_name)


# last fc configuration
if model_type == "res":
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
elif model_type == "vgg":
    num_ftrs = model_ft.classifier[-1].in_features
    model_ft.classifier[-1] = nn.Linear(num_ftrs, num_classes)
elif model_type == "eff":
    num_ftrs = model_ft._fc.in_features
    model_ft._fc = nn.Linear(num_ftrs, num_classes)
elif model_type == "den":
    num_ftrs = model_ft.classifier.in_features
    model_ft._fc = nn.Linear(num_ftrs, num_classes)


## from torchsummary import summary
## summary(model_ft, (3, 224, 224))

## from torchsummaryX import summary
## summary(model_ft, torch.zeros((1, 3, 224, 224)))


model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()


if hp_opt == "sgd":
    optimizer_ft = optim.SGD(
        model_ft.parameters(), lr=hp_lr, momentum=0.9, weight_decay=hp_wd
    )
if hp_opt == "adam":
    optimizer_ft = optim.Adam(
        model_ft.parameters(), lr=hp_lr, weight_decay=hp_wd
    )

if hp_sch == "msl":
    hp_lr_decay_ratio = 0.2

    scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft,
        milestones=[
            hp_ep * 0.3,
            hp_ep * 0.6,
            hp_ep * 0.8,
        ],
        gamma=hp_lr_decay_ratio,
    )
if hp_sch == "cos":
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=hp_ep)


if opt.mode == "grad":
    ## save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    dydi_np_cum = np.zeros((224, 224))

    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)

        x = inputs
        x.requires_grad = True

        y = model_ft(inputs)  # [1, 10]
        # targets.size()  # [1]  # not one hot
        y_c = y[:, targets].sum()  # [1, 1]  # [1, 1]

        y_c.backward(retain_graph=True)
        dydi = inputs.grad

        dydi_np = dydi.cpu().data.numpy()  # 1, 3, 32, 32
        ## dydi_np = dydi_np.sum(axis=(0, 1))
        dydi_np = dydi_np.mean(axis=(0, 1))  # channel mean이니까
        dydi_np = np.maximum(dydi_np, 0)  # pass rleu

        dydi_np_cum += dydi_np


    ### additional
    # min max norm 이전의 값에서 erf 값 찍어보기
    # (224, 224)라서 바로 찍으면 됨
    if True:
        print(dydi_np_cum[100, 100])
        print(dydi_np_cum[100, 101])
        print(dydi_np_cum[100, 102])
        print(dydi_np_cum[100, 103])
        print(dydi_np_cum[100, 104])
        print("")

        print(dydi_np_cum[101, 100])
        print(dydi_np_cum[101, 101])
        print(dydi_np_cum[101, 102])
        print(dydi_np_cum[101, 103])
        print(dydi_np_cum[101, 104])
        print("")

        print(dydi_np_cum[102, 100])
        print(dydi_np_cum[102, 101])
        print(dydi_np_cum[102, 102])
        print(dydi_np_cum[102, 103])
        print(dydi_np_cum[102, 104])
        print("")

        print(dydi_np_cum[103, 100])
        print(dydi_np_cum[103, 101])
        print(dydi_np_cum[103, 102])
        print(dydi_np_cum[103, 103])
        print(dydi_np_cum[103, 104])
        print("")

        print(dydi_np_cum[104, 100])
        print(dydi_np_cum[104, 101])
        print(dydi_np_cum[104, 102])
        print(dydi_np_cum[104, 103])
        print(dydi_np_cum[104, 104])
        print("")

    if False:
        print(dydi_np_cum[100, 100])
        print(dydi_np_cum[100, 101])
        print(dydi_np_cum[100, 102])
        print(dydi_np_cum[100, 103])
        print(dydi_np_cum[100, 104])
        print(dydi_np_cum[100, 105])
        print(dydi_np_cum[100, 106])
        print(dydi_np_cum[100, 107])
        print(dydi_np_cum[100, 108])
        print(dydi_np_cum[100, 109])
        print("")

        print(dydi_np_cum[101, 100])
        print(dydi_np_cum[101, 101])
        print(dydi_np_cum[101, 102])
        print(dydi_np_cum[101, 103])
        print(dydi_np_cum[101, 104])
        print(dydi_np_cum[101, 105])
        print(dydi_np_cum[101, 106])
        print(dydi_np_cum[101, 107])
        print(dydi_np_cum[101, 108])
        print(dydi_np_cum[101, 109])
        print("")

        print(dydi_np_cum[102, 100])
        print(dydi_np_cum[102, 101])
        print(dydi_np_cum[102, 102])
        print(dydi_np_cum[102, 103])
        print(dydi_np_cum[102, 104])
        print(dydi_np_cum[102, 105])
        print(dydi_np_cum[102, 106])
        print(dydi_np_cum[102, 107])
        print(dydi_np_cum[102, 108])
        print(dydi_np_cum[102, 109])
        print("")

        print(dydi_np_cum[103, 100])
        print(dydi_np_cum[103, 101])
        print(dydi_np_cum[103, 102])
        print(dydi_np_cum[103, 103])
        print(dydi_np_cum[103, 104])
        print(dydi_np_cum[103, 105])
        print(dydi_np_cum[103, 106])
        print(dydi_np_cum[103, 107])
        print(dydi_np_cum[103, 108])
        print(dydi_np_cum[103, 109])
        print("")

        print(dydi_np_cum[104, 100])
        print(dydi_np_cum[104, 101])
        print(dydi_np_cum[104, 102])
        print(dydi_np_cum[104, 103])
        print(dydi_np_cum[104, 104])
        print(dydi_np_cum[104, 105])
        print(dydi_np_cum[104, 106])
        print(dydi_np_cum[104, 107])
        print(dydi_np_cum[104, 108])
        print(dydi_np_cum[104, 109])
        print("")

        print(dydi_np_cum[105, 100])
        print(dydi_np_cum[105, 101])
        print(dydi_np_cum[105, 102])
        print(dydi_np_cum[105, 103])
        print(dydi_np_cum[105, 104])
        print(dydi_np_cum[105, 105])
        print(dydi_np_cum[105, 106])
        print(dydi_np_cum[105, 107])
        print(dydi_np_cum[105, 108])
        print(dydi_np_cum[105, 109])
        print("")

        print(dydi_np_cum[106, 100])
        print(dydi_np_cum[106, 101])
        print(dydi_np_cum[106, 102])
        print(dydi_np_cum[106, 103])
        print(dydi_np_cum[106, 104])
        print(dydi_np_cum[106, 105])
        print(dydi_np_cum[106, 106])
        print(dydi_np_cum[106, 107])
        print(dydi_np_cum[106, 108])
        print(dydi_np_cum[106, 109])
        print("")

        print(dydi_np_cum[107, 100])
        print(dydi_np_cum[107, 101])
        print(dydi_np_cum[107, 102])
        print(dydi_np_cum[107, 103])
        print(dydi_np_cum[107, 104])
        print(dydi_np_cum[107, 105])
        print(dydi_np_cum[107, 106])
        print(dydi_np_cum[107, 107])
        print(dydi_np_cum[107, 108])
        print(dydi_np_cum[107, 109])
        print("")

        print(dydi_np_cum[108, 100])
        print(dydi_np_cum[108, 101])
        print(dydi_np_cum[108, 102])
        print(dydi_np_cum[108, 103])
        print(dydi_np_cum[108, 104])
        print(dydi_np_cum[108, 105])
        print(dydi_np_cum[108, 106])
        print(dydi_np_cum[108, 107])
        print(dydi_np_cum[108, 108])
        print(dydi_np_cum[108, 109])
        print("")

        print(dydi_np_cum[109, 100])
        print(dydi_np_cum[109, 101])
        print(dydi_np_cum[109, 102])
        print(dydi_np_cum[109, 103])
        print(dydi_np_cum[109, 104])
        print(dydi_np_cum[109, 105])
        print(dydi_np_cum[109, 106])
        print(dydi_np_cum[109, 107])
        print(dydi_np_cum[109, 108])
        print(dydi_np_cum[109, 109])
        print("")
    ###

    # get first diff index
    x = dydi_np_cum
    x_dx = np.diff(x, axis=0)  # 223, 224
    x_dy = np.diff(x, axis=1)  # 224, 223
    first_diff = (np.mean(np.abs(x_dx)) + np.mean(np.abs(x_dy))) * 0.5

    # get second diff index
    x_ddx = np.diff(x_dx, axis=0)  # 222, 224
    x_ddy = np.diff(x_dy, axis=1)
    second_diff = (np.mean(np.abs(x_ddx)) + np.mean(np.abs(x_ddy))) * 0.5

    print(first_diff)
    print(second_diff)

    # save png
    dydi_np_cum -= np.min(dydi_np_cum)
    dydi_np_cum = np.uint8(dydi_np_cum * 255.0 / np.max(dydi_np_cum))
    cv2.imwrite(save_dir + ".." + "/erf_dydi_caltech101_train_" + model_name + ".png", dydi_np_cum)

if opt.mode == "grad_dAdi":

    ## save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    dAdi_np_cum = np.zeros((224, 224))


    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)

        # hooker funciton
        def save_gradient(grad):
            return grad

        feature_dict = {}

        # img도 hook
        x = inputs
        x.requires_grad = True
        x.register_hook(save_gradient)
        feature_dict["img"] = x

        # for ResNet
        for name, module in model_ft._modules.items():
            # if name != "fc":
            if name not in ["fc", "classifier"]:
                x = module(x)
                x.register_hook(save_gradient)
                feature_dict[name] = x

        if model_type == "res":
            A_ijk = feature_dict["layer4"]  # 1, 512, 7, 7
        elif model_type == "vgg":
            A_ijk = feature_dict["features"]  # 1, 512, 7, 7

        A = A_ijk[0, :, 3, 3].mean()  # channel mean, center가 3, 3.
        A.backward(retain_graph=True)

        dAdi = inputs.grad  # 1, 3, 224, 224
        dAdi_np = dAdi.cpu().data.numpy()
        ## dAdi_np = dAdi_np.sum(axis=(0, 1))
        dAdi_np = dAdi_np.mean(axis=(0, 1))  # channel mean이니까
        dAdi_np = np.maximum(dAdi_np, 0)  # 224, 224
        dAdi_np_cum += dAdi_np


    # for lmfit
    def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):
        # unpack 1D list into 2D x and y coords
        (x, y) = xy_mesh

        # make the 2D Gaussian matrix
        gauss = amp*np.exp(-((x-xc)**2/(2*sigma_x**2)+(y-yc)**2/(2*sigma_y**2)))/(2*np.pi*sigma_x*sigma_y)

        # flatten the 2D Gaussian down to 1D
        return np.ravel(gauss)


    dAdi_np_cum = dAdi_np_cum / np.sum(dAdi_np_cum)

    x = np.arange(224)
    y = np.arange(224)
    xy_mesh = np.meshgrid(x, y)
    noise = dAdi_np_cum
    amp = 1
    xc, yc = np.median(x), np.median(y)
    sigma_x, sigma_y = x[-1] / 10, y[-1] / 6
    guess_vals = [amp * 2, xc * 0.8, yc * 0.8, sigma_x / 1.5, sigma_y / 1.5]
    # tell LMFIT what fn you want to fit, then fit, starting iteration with guess values
    lmfit_model = Model(gaussian_2d)
    lmfit_result = lmfit_model.fit(
        np.ravel(noise),
        xy_mesh=xy_mesh,
        amp=guess_vals[0],
        xc=guess_vals[1],
        yc=guess_vals[2],
        sigma_x=guess_vals[3],
        sigma_y=guess_vals[4]
    )
    # again, calculate R-squared
    lmfit_Rsquared = 1 - lmfit_result.residual.var() / np.var(noise)
    print("Fit R-squared:", lmfit_Rsquared, "\n")
    print(lmfit_result.fit_report())

    # save png
    dAdi_np_cum -= np.min(dAdi_np_cum)
    dAdi_np_cum = np.uint8(dAdi_np_cum * 255.0 / np.max(dAdi_np_cum))
    cv2.imwrite(save_dir + ".." + "/erf_dAdi_caltech101_train_" + model_name + ".png", dAdi_np_cum)


if opt.mode == "grad_dAdi_layer3":

    ## save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    dAdi_np_cum = np.zeros((224, 224))


    # for batch_idx, (inputs, targets) in enumerate(train_loader):
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)

        # hooker funciton
        def save_gradient(grad):
            return grad

        feature_dict = {}

        # img도 hook
        x = inputs
        x.requires_grad = True
        x.register_hook(save_gradient)
        feature_dict["img"] = x

        # for ResNet
        for name, module in model_ft._modules.items():
            # if name != "fc":
            if name not in ["fc", "classifier"]:
                x = module(x)
                x.register_hook(save_gradient)
                feature_dict[name] = x

        if model_type == "res":
            ## A_ijk = feature_dict["layer4"]  # 1, 512, 7, 7
            A_ijk = feature_dict["layer3"]
        # elif model_type == "vgg":
        #     A_ijk = feature_dict["features"]  # 1, 512, 7, 7

        ## A = A_ijk[0, :, 3, 3].mean()  # channel mean, center가 3, 3.
        A = A_ijk[0, :, 7, 7].mean()
        A.backward(retain_graph=True)

        dAdi = inputs.grad  # 1, 3, 224, 224
        dAdi_np = dAdi.cpu().data.numpy()
        ## dAdi_np = dAdi_np.sum(axis=(0, 1))
        dAdi_np = dAdi_np.mean(axis=(0, 1))  # channel mean이니까
        dAdi_np = np.maximum(dAdi_np, 0)  # 224, 224
        dAdi_np_cum += dAdi_np


    # for lmfit
    def gaussian_2d(xy_mesh, amp, xc, yc, sigma_x, sigma_y):
        # unpack 1D list into 2D x and y coords
        (x, y) = xy_mesh

        # make the 2D Gaussian matrix
        gauss = amp*np.exp(-((x-xc)**2/(2*sigma_x**2)+(y-yc)**2/(2*sigma_y**2)))/(2*np.pi*sigma_x*sigma_y)

        # flatten the 2D Gaussian down to 1D
        return np.ravel(gauss)


    dAdi_np_cum = dAdi_np_cum / np.sum(dAdi_np_cum)

    x = np.arange(224)
    y = np.arange(224)
    xy_mesh = np.meshgrid(x, y)
    noise = dAdi_np_cum
    amp = 1
    xc, yc = np.median(x), np.median(y)
    sigma_x, sigma_y = x[-1] / 10, y[-1] / 6
    guess_vals = [amp * 2, xc * 0.8, yc * 0.8, sigma_x / 1.5, sigma_y / 1.5]
    # tell LMFIT what fn you want to fit, then fit, starting iteration with guess values
    lmfit_model = Model(gaussian_2d)
    lmfit_result = lmfit_model.fit(
        np.ravel(noise),
        xy_mesh=xy_mesh,
        amp=guess_vals[0],
        xc=guess_vals[1],
        yc=guess_vals[2],
        sigma_x=guess_vals[3],
        sigma_y=guess_vals[4]
    )
    # again, calculate R-squared
    lmfit_Rsquared = 1 - lmfit_result.residual.var() / np.var(noise)
    print("Fit R-squared:", lmfit_Rsquared, "\n")
    print(lmfit_result.fit_report())

    # save png
    dAdi_np_cum -= np.min(dAdi_np_cum)
    dAdi_np_cum = np.uint8(dAdi_np_cum * 255.0 / np.max(dAdi_np_cum))
    cv2.imwrite(save_dir + ".." + "/erf_dAdi_layer3_caltech101_train_" + model_name + ".png", dAdi_np_cum)

if opt.mode == "train":
    # return이 best model임
    model_ft, best_acc, best_loss = train_model(
        model_ft, criterion, optimizer_ft, scheduler, num_epochs=hp_ep
    )

    # finally, save the best
    os.makedirs(save_dir, exist_ok=True)  # 폴더가 존재하지 않으면 생성하고, 존재하면 아무것도 하지 않도록.
    save_dirfile = save_dir + "/ba_" + str(best_acc)[:6] + "_bl_" + str(best_loss)[:6]
    torch.save(model_ft.state_dict(), save_dirfile)


if opt.mode == "train_micro_object":
    # return이 best model임
    model_ft, best_acc, best_loss = train_model_micro_object(
        model_ft, criterion, optimizer_ft, scheduler, num_epochs=hp_ep
    )

    # finally, save the best
    os.makedirs(save_dir, exist_ok=True)  # 폴더가 존재하지 않으면 생성하고, 존재하면 아무것도 하지 않도록.
    save_dirfile = save_dir + "/ba_" + str(best_acc)[:6] + "_bl_" + str(best_loss)[:6]
    torch.save(model_ft.state_dict(), save_dirfile)

if opt.mode == "train_micro_object_rgb":
    # return이 best model임
    model_ft, best_acc, best_loss = train_model_micro_object_rgbint(
        model_ft, criterion, optimizer_ft, scheduler, num_epochs=hp_ep
    )

    # finally, save the best
    os.makedirs(save_dir, exist_ok=True)  # 폴더가 존재하지 않으면 생성하고, 존재하면 아무것도 하지 않도록.
    save_dirfile = save_dir + "/ba_" + str(best_acc)[:6] + "_bl_" + str(best_loss)[:6]
    torch.save(model_ft.state_dict(), save_dirfile)

if opt.mode == "train_micro_object_rgb_rgb":
    # return이 best model임
    model_ft, best_acc, best_loss = train_model_micro_object_rgbint_rgb(
        model_ft, criterion, optimizer_ft, scheduler, num_epochs=hp_ep
    )

    # finally, save the best
    os.makedirs(save_dir, exist_ok=True)  # 폴더가 존재하지 않으면 생성하고, 존재하면 아무것도 하지 않도록.
    save_dirfile = save_dir + "/ba_" + str(best_acc)[:6] + "_bl_" + str(best_loss)[:6]
    torch.save(model_ft.state_dict(), save_dirfile)

if opt.mode == "train_micro_mnist":
    # 사실 train은 그대로고, dataset만 약간 변형해서 하는 게 맞을듯. 정확하게는, loader를 잘 설계하면 됨.

    model_ft, best_acc, best_loss = train_model_micro_mnist(
        model_ft, criterion, optimizer_ft, scheduler, num_epochs=hp_ep
    )

    # no save ok


if opt.mode == "attack":
    save_dir = save_dir + "/sample/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    noise_list = []
    fail_index_list = []
    num_fail = 0
    num_iter = 1000
    hp_lr_attack = 1e+1
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # print("")
        # print(batch_idx)
        # inputs, targets = next(iter(test_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        y = model_ft(inputs)  # [1, 10]
        # print(inputs.size())  # 1, 3, 224, 224

        # define attack
        attack_noise = torch.nn.Parameter(torch.randn(1, requires_grad=True, device=device))
        attack_x = opt.hp_ax
        attack_y = opt.hp_ay

        optimizer = optim.Adam([attack_noise], lr=hp_lr_attack)
        is_success = False

        for i in range(num_iter):
            attack_zeros = torch.zeros_like(inputs)
            attack_zeros[:, :, attack_x, attack_y] += attack_noise
            inputs_attack = inputs + attack_zeros
            y_attack = model_ft(inputs_attack)  # [1, 10]
            mse = -torch.mean((y - y_attack)**2)
            mse.backward(retain_graph=True)
            optimizer.step()

            if (torch.argmax(y) != torch.argmax(y_attack)):
                is_success = True
                break
        if is_success:
            end_noise = torch.abs(attack_noise).item()
            noise_list.append(end_noise)
            # print("success")
            # print(i)
            # print(end_noise)
            # print(sum(noise_list) / len(noise_list))
        else:
            num_fail += 1
            fail_index_list.append(batch_idx)
            # print("fail")
            # print(num_fail)

    # finally,
    # print("success")
    # print(sum(noise_list))
    # print(len(noise_list))
    # print(sum(noise_list) / len(noise_list))
    # print("")
    print("fail")
    print(fail_index_list)
    print(num_fail)
    print("")

if opt.mode == "conf":
    save_dir = save_dir + "/sample/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    conf_drop_cum = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        y = model_ft(inputs)  # [1, 102]

        # define noise
        attack_x = opt.hp_ax
        attack_y = opt.hp_ay
        inputs[:, :, attack_x, attack_y] = 0.
        y_noise = model_ft(inputs)  # [1, 102]

        y = y.cpu().detach()
        y_noise = y_noise.cpu().detach()
        max_indices = torch.argmax(y, axis=1)  # [1], 정수 index임
        y_max = y[:, max_indices]
        y_noise_max = y_noise[:, max_indices]
        conf_drop = torch.maximum(y_max - y_noise_max, torch.zeros(1))
        conf_drop_cum += conf_drop

    print(conf_drop_cum)


if opt.mode == "translate":
    save_dir = save_dir + "/sample/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    translation_diff_cum_sum_val = 0.
    for batch_idx, (inputs, targets) in enumerate(val_loader_trans):
        # print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.size())  # 32, 3, 225, 225
        inputs_t1 = inputs[:, :, :224, :224]  # 32, 3, 224, 224
        # inputs_t2 = inputs[:, :, 1:, 1:]  # 32, 3, 224, 224
        inputs_t2 = inputs[:, :, hp_tr:, hp_tr:]  # 32, 3, 224, 224

        # model in
        y1 = model_ft(inputs_t1)  # 임의의 실수 같긴함. 아마 logit.
        y2 = model_ft(inputs_t2)

        translation_diff_cum_sum_val += torch.sum(torch.abs(y1 - y2)).item()
    print(translation_diff_cum_sum_val)

    translation_diff_cum_sum_test = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader_trans):
        # print(batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        # print(inputs.size())  # 32, 3, 225, 225
        inputs_t1 = inputs[:, :, :224, :224]  # 32, 3, 224, 224
        # inputs_t2 = inputs[:, :, 1:, 1:]  # 32, 3, 224, 224
        inputs_t2 = inputs[:, :, hp_tr:, hp_tr:]  # 32, 3, 224, 224

        # model in
        y1 = model_ft(inputs_t1)  # 임의의 실수 같긴함. 아마 logit.
        y2 = model_ft(inputs_t2)

        translation_diff_cum_sum_test += torch.sum(torch.abs(y1 - y2)).item()
    print(translation_diff_cum_sum_test)


if opt.mode == "translate_micro":
    save_dir = save_dir + "/sample/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    # val
    cum_sum_val = 0.
    cum_count_val = 0.
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs_micro = inputs.clone().detach()
        inputs_micro = torch.stack([put_micro_object_at(input_single, hp_lx, hp_ly) for input_single in inputs_micro])  # 123, 3, 224, 224
        inputs_micro = inputs_micro.to(device)

        y_micro = model_ft(inputs_micro)  # 128, 2
        cum_sum_val += torch.sum(y_micro[:, 1]).item()
        _, preds = torch.max(y_micro, 1)
        cum_count_val += torch.sum(preds).item()

    print(cum_sum_val)
    print(cum_count_val)

    # test
    cum_sum_test = 0.
    cum_count_test = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs_micro = inputs.clone().detach()
        inputs_micro = torch.stack([put_micro_object_at(input_single, hp_lx, hp_ly) for input_single in inputs_micro])  # 123, 3, 224, 224
        inputs_micro = inputs_micro.to(device)

        y_micro = model_ft(inputs_micro)  # 128, 2
        cum_sum_test += torch.sum(y_micro[:, 1]).item()
        _, preds = torch.max(y_micro, 1)
        cum_count_test += torch.sum(preds).item()

    print(cum_sum_test)
    print(cum_count_test)


if opt.mode == "noise_accuracy":
    # save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    running_corrects = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # define image with noise
        attack_x = opt.hp_ax
        attack_y = opt.hp_ay
        # inputs[:, :, attack_x, attack_y] += opt.hp_no
        inputs[:, :, attack_x:attack_x+hp_le, attack_y:attack_y+hp_le] += opt.hp_no

        # get output
        outputs = model_ft(inputs)  # [1, 102]
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == targets.data)

    epoch_acc = running_corrects.double() / dataset_sizes["test"]
    # print("{:4f}".format(epoch_acc))
    print("{:.8f}".format(epoch_acc))

if opt.mode == "noise_sqdiff":
    # save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    sqdiff_cum = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        y = model_ft(inputs)  # [1, 102]
        y = y.cpu().detach()

        # define image with noise
        attack_x = opt.hp_ax
        attack_y = opt.hp_ay
        # inputs[:, :, attack_x, attack_y] += opt.hp_no
        inputs[:, :, attack_x:attack_x+hp_le, attack_y:attack_y+hp_le] += opt.hp_no
        y_noise = model_ft(inputs)  # [1, 102]
        y_noise = y_noise.cpu().detach()
        sqdiff = torch.sum((y - y_noise)**2) / num_classes

        sqdiff_cum += sqdiff

    # print("{:4f}".format(sqdiff_cum))
    print("{:.8f}".format(sqdiff_cum))

if opt.mode == "noise_absdiff":
    # save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    absdiff_cum = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        y = model_ft(inputs)  # [1, 102]
        y = y.cpu().detach()

        # define image with noise
        attack_x = opt.hp_ax
        attack_y = opt.hp_ay
        # inputs[:, :, attack_x, attack_y] += opt.hp_no
        inputs[:, :, attack_x:attack_x+hp_le, attack_y:attack_y+hp_le] += opt.hp_no
        y_noise = model_ft(inputs)  # [1, 102]
        y_noise = y_noise.cpu().detach()
        absdiff = torch.sum(torch.abs(y - y_noise)) / num_classes

        absdiff_cum += absdiff

    print("{:.8f}".format(absdiff_cum))

if opt.mode == "noise_acmdrop":
    # save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    acmdrop_cum = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        y = model_ft(inputs)  # [1, 102]
        y = y.cpu().detach()

        # define image with noise
        attack_x = opt.hp_ax
        attack_y = opt.hp_ay
        # inputs[:, :, attack_x, attack_y] += opt.hp_no
        inputs[:, :, attack_x:attack_x+hp_le, attack_y:attack_y+hp_le] += opt.hp_no
        y_noise = model_ft(inputs)  # [1, 102]
        y_noise = y_noise.cpu().detach()
        # absdiff = torch.sum(torch.abs(y - y_noise)) / num_classes
        acmdrop = torch.sum(torch.maximum(y - y_noise, torch.zeros_like(y))) / num_classes

        acmdrop_cum += acmdrop

    print("{:.8f}".format(acmdrop_cum))

if opt.mode == "noise_confdrop":
    # save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    conf_drop_cum = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        y = model_ft(inputs)  # [1, 102]
        y = y.cpu().detach()

        # define image with noise
        attack_x = opt.hp_ax
        attack_y = opt.hp_ay
        # inputs[:, :, attack_x, attack_y] += opt.hp_no
        inputs[:, :, attack_x:attack_x+hp_le, attack_y:attack_y+hp_le] += opt.hp_no
        y_noise = model_ft(inputs)  # [1, 102]
        y_noise = y_noise.cpu().detach()

        # y_conf = y[:, targets]  # [1, 1]
        # y_noise_conf = y_noise[:, targets]  # [1, 1]

        drop = (y - y_noise)[:, targets]  # [1, 1]
        conf_drop = torch.maximum(drop[0, 0], torch.zeros(1))  # [1]
        conf_drop_cum += conf_drop

    # print("{:4f}".format(conf_drop_cum[0]))
    print("{:.8f}".format(conf_drop_cum[0]))

if opt.mode == "noise_confdropbatch":
    # save_dir = save_dir + "/sample/"
    save_dir = "main_sample/" + save_dir + "/"
    save_file = os.listdir(save_dir)[0]
    save_dirfile = save_dir + save_file
    model_ft.load_state_dict(torch.load(save_dirfile))

    model_ft.eval()

    acmdrop_cum = 0.
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        y = model_ft(inputs)  # [1, 102]
        y = y.cpu().detach()

        # define image with noise
        attack_x = opt.hp_ax
        attack_y = opt.hp_ay
        # inputs[:, :, attack_x, attack_y] += opt.hp_no
        inputs[:, :, attack_x:attack_x+hp_le, attack_y:attack_y+hp_le] += opt.hp_no
        y_noise = model_ft(inputs)  # [64, 37]
        y_noise = y_noise.cpu().detach()

        # y_conf = y[:, targets]  # [1, 1]
        # y_noise_conf = y_noise[:, targets]  # [1, 1]

        # extract only corresponding class
        y = torch.stack([y_elem[targets_elem] for y_elem, targets_elem in zip(y, targets)])  # [64]
        y_noise = torch.stack([y_elem[targets_elem] for y_elem, targets_elem in zip(y_noise, targets)])

        acmdrop = torch.sum(torch.maximum(y - y_noise, torch.zeros_like(y))) / num_classes
        acmdrop_cum += acmdrop

    print("{:.8f}".format(acmdrop_cum))

