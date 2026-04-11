import os
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from nupic.torch.modules import KWinners2d, rezero_weights, update_boost_strength
from torch.utils.data import random_split
from torch.utils.data import random_split, TensorDataset, DataLoader
from torch.utils.data import ConcatDataset
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from torch.utils.data import SubsetRandomSampler

seed_val = 1
torch.manual_seed(seed_val)
np.random.seed(seed_val)
LEARNING_RATE = 0.01  # Recommend 0.01
MOMENTUM = 0.5
EPOCHS = 10  # Recommend 10
FIRST_EPOCH_BATCH_SIZE = 4  # Used for optimizing k-WTA
TRAIN_BATCH_SIZE = 128  # Recommend 128
TEST_BATCH_SIZE = 512
PERCENT_ON = 0.15  # Recommend 0.15
BOOST_STRENGTH = 20.0  # Recommend 20
DATASET = "mnist"  # Options are "mnist" or "fashion_mnist"; note in some cases
#==================================================================================
GRID_SIZE = [5] # <-----------
NUM_CLASSES = 10 # < --------------


class SDRCNNBase_(nn.Module):
    def __init__(self, percent_on=0.15, boost_strength=20.0, grid=5):
        self.grid = grid
        super(SDRCNNBase_, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, padding=0)# k=5, p=2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=0) # k=5, p=0
        self.pool2 = nn.AdaptiveMaxPool2d((grid, grid))
        self.k_winner = KWinners2d(channels=128, percent_on=percent_on, boost_strength=boost_strength, local=True)
        self.dense1 = nn.Linear(in_features=128 * grid**2, out_features=256)
        self.dense2 = nn.Linear(in_features=256, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=NUM_CLASSES) # CHANGED HERE
        self.softmax = nn.LogSoftmax(dim=1)
        self.flatten = nn.Flatten()

    def until_kwta(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.k_winner(x)
        return x

    def forward(self, inputs):
        x = self.until_kwta(inputs)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        return x

    def output_sdr(self, inputs):
        x = self.until_kwta(inputs)
        x = (x > 0).float()
        x = x.reshape((-1, 128, self.grid**2))
        x = x.permute(0, 2, 1)
        return x


def post_batch(model):
    model.apply(rezero_weights)


def train(model, loader, optimizer, criterion, post_batch_callback=None):

    model.train()
    for data, target in tqdm(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if post_batch_callback is not None:
            post_batch(model)


def test(model, loader, criterion, save_name=None, grid_size=5):
    model.eval()
    loss = 0
    total_correct = 0
    all_sdrs = []
    all_labels = []

    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            all_sdrs.append(np.asarray(model.output_sdr(data), dtype=np.int8)) # output_sdr.shape = patch, dim
            all_labels.append(target)

            loss += target.size(0) * criterion(output, target,).item()  # sum up batch
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max
            total_correct += pred.eq(target.view_as(pred)).sum().item()

    all_sdrs = np.concatenate(all_sdrs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if save_name is not None:
        name = str(NUM_CLASSES) + '-' + str(grid_size)
        np.save("python2_htm_docker/docker_dir/dataset/" + name + "_vecs_" + save_name, all_sdrs)
        np.save("python2_htm_docker/docker_dir/dataset/" + name + "_labels_" + save_name, all_labels)

    return {"accuracy": total_correct / len(loader.dataset),
            "loss": loss / len(loader.dataset),
            "total_correct": total_correct}


class RelabelImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        self.class_to_true_label = {0: 10, 1: 11}
        #self.class_to_true_label = {i: int(cls_name) for cls_name, i in self.class_to_idx.items()}

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        true_label = self.class_to_true_label[label]
        return image, true_label


def data_setup10():
    train_all = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    #train_all = Subset(train_all, list(range(512*10)))
    test = datasets.MNIST('data', train=False, download=True, transform=transforms.ToTensor())
    #test = Subset(test, list(range(512*2)))
    n_train = int(0.8 * len(train_all))
    n_val = len(train_all) - n_train
    train, valid = random_split(train_all, [n_train, n_val])

    # datasetalize
    first_loader = DataLoader(train, batch_size=FIRST_EPOCH_BATCH_SIZE)
    train_loader = DataLoader(train, batch_size=TRAIN_BATCH_SIZE)
    valid_loader = DataLoader(valid, batch_size=TEST_BATCH_SIZE)
    test_loader = DataLoader(test, batch_size=TEST_BATCH_SIZE)

    return first_loader, train_loader, valid_loader, test_loader


if __name__ == "__main__":
    first_l, train_l, valid_l, test_l = data_setup10()

    for grid_size in GRID_SIZE:

        model = SDRCNNBase_()
        sgd = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
        xent = nn.CrossEntropyLoss()

        # boost-strength
        print('boost strength')
        train(model, loader=first_l, optimizer=sgd, criterion=xent, post_batch_callback=True)
        model.apply(update_boost_strength)

        # training
        print('training')
        for epoch in range(EPOCHS):
            train(model, loader=train_l, optimizer=sgd, criterion=xent, post_batch_callback=True)
            model.apply(update_boost_strength)
            test(model, test_l, criterion=xent,)
        torch.save(model.state_dict(), 'saved_networks/10-5.pt')

        # save for training GCN
        print('saving for training GCN')
        test(model, valid_l, criterion=xent, save_name='training')

        # save for testing GCN
        print('saving for testing GCN')
        test(model, test_l, criterion=xent, save_name='testing')
