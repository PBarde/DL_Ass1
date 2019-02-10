from __future__ import print_function

import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np
import torchvision.transforms
import random
import os
from archi import *
from torch.autograd import Variable

from torch.utils.data.sampler import SubsetRandomSampler

from_numpy = torch.from_numpy

## Load data and splits it between train and test sets
SEED = 131214
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

data_size = (3, 64, 64)
batch_size = 64
classes_dict = {0: 'Cat', 1: 'Dog'}
distrib_means = (0.48973373, 0.45465815, 0.4159738)
distrib_stds = (0.25206217, 0.24510814, 0.24726307)
data_augmentation = True
l2_coeff = 5e-4

# balanced training set as many cats than dogs

base_transforms = [torchvision.transforms.ToTensor(),
                   torchvision.transforms.Normalize(distrib_means, distrib_stds)]
r_choice = torchvision.transforms.RandomChoice
compose = torchvision.transforms.Compose

if data_augmentation:
    augmentations = [torchvision.transforms.RandomHorizontalFlip(),
                     torchvision.transforms.RandomResizedCrop(64),
                     torchvision.transforms.RandomRotation(90),
                     torchvision.transforms.RandomVerticalFlip()]

    # aug_transforms = compose(
    #     [r_choice(augmentations +
    #               [torchvision.transforms.RandomOrder(augmentations)])]
    #     + base_transforms)
    aug_transforms = compose(
        [torchvision.transforms.RandomApply([r_choice(augmentations +
                                                      [torchvision.transforms.RandomOrder(augmentations)])],
                                            p=0.75)]
        + base_transforms)
else:
    aug_transforms = compose(base_transforms)

base_transforms = compose(base_transforms)

train_data = torchvision.datasets.ImageFolder(root='./trainset',
                                              transform=aug_transforms)

test_data = torchvision.datasets.ImageFolder(root='./trainset',
                                             transform=base_transforms)

len_data = len(train_data)
# Creating data indices for training and validation splits:
indices = list(range(len_data))
id_split = int(0.9 * len_data)
np.random.shuffle(indices)
train_indices, test_indices = indices[:id_split], indices[id_split:]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
test_sampler = SubsetRandomSampler(test_indices)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, sampler=train_sampler, num_workers=2,
    worker_init_fn=random.seed(SEED))

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, sampler=test_sampler, num_workers=2,
    worker_init_fn=random.seed(SEED))

# building model
model_type = 'CNN'
cuda = torch.cuda.is_available()

if cuda:
    print('cuda is available')
else:
    print('cuda is not available')

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
elif model_type == 'ResNet':
    model = CIFARResNet18(num_classes=2)

else:
    raise ValueError

if cuda:
    model = model.cuda()

# summary(model,data_size)
summary(model, (3, 64, 64))

## Setting the optimizer
num_epochs = 200  # number of training epochs
lr0 = 0.1
criterion = nn.CrossEntropyLoss()  # to compute the loss
optimizer = optim.SGD(model.parameters(), lr=lr0)
# lr_lambda = lambda epoch: 0.1**(epoch/float(num_epochs))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, mode='max')


## Defining the evaluation routines
def accuracy(proba, y):
    correct = torch.eq(proba.max(1)[1], y).sum().type(torch.FloatTensor)
    return correct / y.size(0)


def evaluate(dataset_loader, criterion):
    LOSSES = 0
    COUNTER = 0
    for batch in dataset_loader:
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

        loss = criterion(model(x), y)
        n = y.size(0)
        LOSSES += loss.sum().data.cpu().numpy() * n
        COUNTER += n

    return LOSSES / float(COUNTER)


def L2_loss(coeff):
    l = Variable(torch.FloatTensor(1), requires_grad=True).cuda()
    for w in model.named_parameters():
        if 'weight' in w[0]:
            l = l + 0.5 * torch.pow(w[1], 2).sum()
    return l * coeff

## Defines the train function
def train_model():
    root_path = './res_L2/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    learning_curve_nll_train = list()
    learning_curve_nll_test = list()
    learning_curve_acc_train = list()
    learning_curve_acc_test = list()
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

            loss = criterion(model(x), y)

            if l2_coeff is not None:
                loss = loss + L2_loss(l2_coeff)

            loss.backward()
            optimizer.step()

            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            COUNTER += n
            ITERATIONS += 1
            if ITERATIONS % (store_every / 5) == 0:
                avg_loss = LOSSES / float(COUNTER)
                LOSSES = 0
                COUNTER = 0
                print(" Iteration {}: TRAIN {}".format(
                    ITERATIONS, avg_loss))

            if ITERATIONS % (store_every) == 0:

                train_loss = evaluate(train_loader, criterion)
                learning_curve_nll_train.append(train_loss)
                test_loss = evaluate(test_loader, criterion)
                learning_curve_nll_test.append(test_loss)

                train_acc = evaluate(train_loader, accuracy)
                learning_curve_acc_train.append(train_acc)
                test_acc = evaluate(test_loader, accuracy)
                learning_curve_acc_test.append(test_acc)
                if round(test_acc, 3) > best_acc:
                    best_acc = round(test_acc, 3)
                    path_to_best = root_path + 'd_aug' * int(data_augmentation) + f'_test_acc_{best_acc}.pth'
                    torch.save(model, path_to_best)
                    print('saved model')

                print(" [NLL] TRAIN {} / TEST {}".format(
                    train_loss, test_loss))
                print(" [ACC] TRAIN {} / TEST {}".format(
                    train_acc, test_acc))

                scheduler.step(test_loss)

    return learning_curve_nll_train, \
           learning_curve_nll_test, \
           learning_curve_acc_train, \
           learning_curve_acc_test, path_to_best


store_every = 1000
nll_train, nll_test, acc_train, acc_test, p_best = train_model()
# import pickle
# fp=open('drive/Colab Notebooks/q3_moredata2/summary.pckl','wb')
# pickle.dump([nll_train, nll_test, acc_train, acc_test, p_best], fp)
