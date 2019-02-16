from __future__ import print_function
import sys
sys.path.insert(0, "./..")
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np
import torchvision.transforms
import random
import os
from P3.archi import *
import pickle

from torch.utils.data.sampler import SubsetRandomSampler

from_numpy = torch.from_numpy

## Load data and splits it between train and test sets
SEED = 131214
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available')
else:
    print('cuda is not available')

data_size = (3, 64, 64)

classes_dict = {0: 'Cat', 1: 'Dog'}
distrib_means = (0.48973373, 0.45465815, 0.4159738)
distrib_stds = (0.25206217, 0.24510814, 0.24726307)

def get_loaders(batch_size, data_augmentation):
    # balanced training set as many cats than dogs
    base_transforms = [torchvision.transforms.ToTensor(),
                       torchvision.transforms.Normalize(distrib_means, distrib_stds)]
    r_choice = torchvision.transforms.RandomChoice
    compose = torchvision.transforms.Compose

    if data_augmentation:
        augmentations = [torchvision.transforms.RandomHorizontalFlip(),
                         torchvision.transforms.RandomResizedCrop(64),
                         torchvision.transforms.RandomRotation(90),
                         torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                         torchvision.transforms.RandomVerticalFlip()]

        aug_transforms = compose(
            [torchvision.transforms.RandomApply([r_choice(augmentations +
                                                          [torchvision.transforms.RandomOrder(augmentations)])],
                                                p=1.)]
            + base_transforms)
    else:
        aug_transforms = compose(base_transforms)

    base_transforms = compose(base_transforms)

    train_data = torchvision.datasets.ImageFolder(root='./trainset',
                                                  transform=aug_transforms)

    valid_data = torchvision.datasets.ImageFolder(root='./trainset',
                                                 transform=base_transforms)

    len_data = len(train_data)
    # Creating data indices for training and validation splits:
    indices = list(range(len_data))
    id_split = int(0.9 * len_data)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[:id_split], indices[id_split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=2,
        worker_init_fn=random.seed(SEED))

    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch_size, sampler=valid_sampler, num_workers=2,
        worker_init_fn=random.seed(SEED))
    return train_loader, valid_loader

## Defining the evaluation routines
def accuracy(proba, y):
    correct = torch.eq(proba.max(1)[1], y).sum().type(torch.FloatTensor)
    return correct / y.size(0)


def evaluate(model, dataset_loader, criterion, model_type):
    LOSSES = 0
    COUNTER = 0
    for batch in dataset_loader:
        x, y = batch
        if model_type == 'MLP':
            x = x.view(-1, 784)
            y = y.view(-1)
        elif model_type == 'CNN':
            x = x.view(-1, *data_size)
            y = y.view(-1)

        if cuda:
            x = x.cuda()
            y = y.cuda()

        loss = criterion(model(x), y)
        n = y.size(0)
        LOSSES += loss.sum().data.cpu().numpy() * n
        COUNTER += n

    return LOSSES / float(COUNTER)


def L2_loss(model, coeff):
    l = torch.tensor(0., requires_grad=True)
    if cuda:
        l = l.cuda()

    for w in model.named_parameters():
        if 'weight' in w[0]:
            l = l + 0.5 * torch.pow(w[1], 2).sum()
    return l * coeff


def build_model(model_type):

    if model_type == 'MLP':
        model = nn.Sequential(
            ResLinear(784, 312),
            nn.ReLU(),
            ResLinear(312, 312),
            nn.ReLU(),
            ResLinear(312, 10)
        )
    elif model_type == 'CNN':
        model = nn.Sequential(
            nn.Conv2d(3, 16, 5),  # 1 input channel, 16 output channel, 5x5 kernel
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            Flatten(),
            ResLinear(2704, 100),
            nn.ReLU(),
            ResLinear(100, 10)
        )
    elif model_type == 'ResNet_3':
        model = CIFARResNet18(num_classes=2, k=3)
    elif model_type == 'ResNet_5':
        model = CIFARRestNet18(num_classes=2, k=5)

    else:
        raise ValueError

    if cuda:
        model = model.cuda()

    return model


## Defines the train function
def train_model(train_loader, valid_loader, model, criterion, optimizer, scheduler, root_path, num_epochs, model_type,
                l2_coeff):
    cuda = torch.cuda.is_available()
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    learning_curve_nll_train = list()
    learning_curve_nll_valid = list()
    learning_curve_acc_train = list()
    learning_curve_acc_valid = list()
    best_acc = -np.inf
    for e in range(num_epochs):
        print(f'============= EPOCH {e} ========================')
        for batch in train_loader:
            optimizer.zero_grad()

            x, y = batch
            if model_type == 'MLP':
                x = x.view(-1, 784)
                y = y.view(-1)
            elif model_type == 'CNN':
                x = x.view(-1, *data_size)
                y = y.view(-1)
            if cuda:
                x = x.cuda()
                y = y.cuda()

            if l2_coeff is not None:
                loss = criterion(model(x), y) + L2_loss(model, l2_coeff)

            else:
                loss = criterion(model(x), y)

            loss.backward()
            optimizer.step()

            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            COUNTER += n
            ITERATIONS += 1

        avg_loss = LOSSES / float(COUNTER)
        LOSSES = 0
        COUNTER = 0
        print(" Epoch {}: TRAIN {}".format(
            e, avg_loss))

        if (e+1.)%store_every == 0:
            train_loss = evaluate(model, train_loader, criterion, model_type)
            learning_curve_nll_train.append(train_loss)
            valid_loss = evaluate(model, valid_loader, criterion, model_type)
            learning_curve_nll_valid.append(valid_loss)

            train_acc = evaluate(model, train_loader, accuracy, model_type)
            learning_curve_acc_train.append(train_acc)
            valid_acc = evaluate(model, valid_loader, accuracy, model_type)
            learning_curve_acc_valid.append(valid_acc)
            if round(valid_acc, 3) > best_acc:
                best_acc = round(valid_acc, 3)
                path_to_best = root_path + 'd_aug' * int(data_augmentation) + f'_valid_acc_{best_acc}.pth'
                torch.save(model, path_to_best)
                print('saved model')

            print(" [NLL] TRAIN {} / valid {}".format(
                train_loss, valid_loss))
            print(" [ACC] TRAIN {} / valid {}".format(
                train_acc, valid_acc))

            scheduler.step(valid_acc)


    fp = open(root_path + 'summary.pckl', 'wb')
    pickle.dump([learning_curve_nll_train, learning_curve_nll_valid,
                 learning_curve_acc_train, learning_curve_acc_valid, path_to_best], fp)

    return learning_curve_nll_train, \
           learning_curve_nll_valid, \
           learning_curve_acc_train, \
           learning_curve_acc_valid,


if __name__== '__main__':
    batch_size = 64
    data_augmentation = True
    model_type = 'ResNet_3'
    train_loader, valid_loader = get_loaders(batch_size, data_augmentation)
    store_every = 3
    lr0 = 0.1
    num_epochs = 200  # number of training epochs
    l2_coeff_list = [5e-4, 1e-4]
    criterion = nn.CrossEntropyLoss()  # to compute the loss

    for l2_coeff in l2_coeff_list:
        root_path = f'./models/'
        model = build_model(model_type)
        optimizer = optim.SGD(model.parameters(), lr=lr0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode='max')
        train_model(train_loader, valid_loader, model, criterion, optimizer, scheduler, root_path,
                    num_epochs, model_type, l2_coeff)


