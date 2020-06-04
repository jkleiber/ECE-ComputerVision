import torch

from fastai import *
from fastai.vision import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.lin1 = nn.Linear(1025, 200)
        self.lin2 = nn.Linear(200, 150)
        self.lin3 = nn.Linear(150, 2)

        # Do Xavier initialization
        nn.init.xavier_normal_(self.lin1.weight)
        nn.init.xavier_normal_(self.lin2.weight)
        nn.init.xavier_normal_(self.lin3.weight)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)
        return x


def process_batch(batch):
    batch_size = batch[0].shape[0]
    inputs=np.array(batch[0])
    inputs=inputs[:,0,:,:] # just take the first color (grayscale image anyway)
    inputs=np.reshape(inputs,(inputs.shape[0],-1))
    inputs=np.concatenate((inputs,np.ones((batch_size,1))), axis=1)
    labels=np.array(batch[1]).transpose()

    # Re-orient the inputs
    return inputs, labels


def calculate_error_rate(batch, predicted, labels):
    correct = 0
    total = batch[0].shape[0]
    for i in range(total):
        correct += (predicted[i] == labels[i]).sum().item()
    err_rate = 1 - correct/total
    return correct, err_rate


if __name__ == "__main__":
    # Load data from MNIST dataset
    mnist = untar_data(URLs.MNIST_TINY)
    tfms = get_transforms(do_flip=False)

    # List the data in training, validation and test sets
    data = (ImageList.from_folder(mnist)
            .split_by_folder()
            .label_from_folder()
            .add_test_folder(f'{mnist}' +'/test/')
            .transform(tfms, size=32)
            .databunch()
            .normalize(imagenet_stats))

    # Show the data
    # data.show_batch()

    # Get the first MNIST batch from fastai
    batch = data.one_batch()
    batch_size, no_channels, dimx, dimy = batch[0].shape

    # Set up the PyTorch model
    net = Network()

    # Set up the optimization system
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005)

    # Reset the gradients
    optimizer.zero_grad()

    # Train the PyTorch model
    for epoch in range(50):  # loop over the dataset multiple times
        x, labels = process_batch(batch)

        # Convert inputs and labels to PyTorch variable
        x = Variable(torch.from_numpy(x).float())
        labels = Variable(torch.from_numpy(labels))

        # Run a forward pass
        outputs = net(x)

        # Compute loss and backpropagate
        loss = criterion(outputs, labels)
        loss.backward()

        # Optimize parameters
        optimizer.step()

        # Calculate error rate
        _, predicted = torch.max(outputs, 1)
        _, err_rate = calculate_error_rate(batch, predicted, labels)

        # Print statistics
        print(f"Cost: {loss.item()}, Error rate: {err_rate}")

        # Get the next MNIST batch from fastai
        batch = data.one_batch()

    print('Training Complete!')

    print('Running on the validation set...')
    num_correct = 0
    total = 0
    for i in range(50):
        # Test the PyTorch model on the validation data
        # Get the test batch
        valid_batch = data.one_batch(DatasetType.Valid)
        x, labels = process_batch(valid_batch)

        # Convert inputs and labels to PyTorch variable
        x = Variable(torch.from_numpy(x).float())
        labels = Variable(torch.from_numpy(labels))

        # Run on the test set and find the error
        outputs = net(x)
        _, predicted = torch.max(outputs, 1)

        # Calculate error rate
        correct, err_rate = calculate_error_rate(valid_batch, predicted, labels)
        num_correct += correct
        total += valid_batch[0].shape[0]

        print(f"Error rate: {err_rate}")

    err_rate = 1 - num_correct / total

    # Display statistics
    print('Validation Results')
    print(f"Total error rate: {err_rate}")
