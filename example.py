'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
    
    Example script training a multilayer perceptron on the MNIST dataset
    demonstrating the PyTorch implementation of the scheduler based on the fluctuation-dissipation relation described in:
    Sho Yaida, "Fluctuation-dissipation relations for stochastic gradient descent," ICLR 2019.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import torchvision.transforms as transforms

from FDR_SGD import FDR_quencher

class MLP(nn.Module):
    '''
        Simple MLP for demonstration.
    '''
    def __init__(self, in_channel=1, im_size=28, num_classes=10, fc_channel1=200, fc_channel2=200):
        super(MLP, self).__init__()
    
        # Parameter setup
        compression=in_channel*im_size*im_size
        self.compression=compression
    
        # Structure
        self.fc1 = nn.Linear(compression, fc_channel1)
        self.fc2 = nn.Linear(fc_channel1, fc_channel2)
        self.fc3 = nn.Linear(fc_channel2, num_classes)
    
        # Initialization protocol
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, self.compression)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    '''
        Train MLP for MNIST dataset with FDR scheduler.
    '''
    # Set up seed and devices
    seed=1
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed(seed)
    else:
        device = torch.device("cpu")

    # Load the MNIST data
    n_batch=100
    downloadFlag=True
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,download=downloadFlag, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=n_batch, shuffle=True, num_workers=1)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=downloadFlag, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=n_batch, shuffle=False, num_workers=1)

    # Initialize the model
    model = MLP()
    model.to(device)

    # Set the loss
    criterion = nn.CrossEntropyLoss()

    '''
        Set the FDR scheduler here
    '''
    optimizer = FDR_quencher(model.parameters(), lr_init=0.1, momentum=0.5, dampening=0.0, weight_decay=0.01, t_adaptive=1000, X=0.01, Y=0.9)

    # Standard Train-Test loop
    for epoch in range(200):
        print('Epoch %d' % (epoch+1) )
        
        # Train
        model.train()
        running_training_loss = 0.0
        for count, (input, target) in enumerate(trainloader):
            input, target = input.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(input) # forward pass
            loss = criterion(output, target) # loss
            loss.backward() # computes gradients
            optimizer.step() # optimizer step
            running_training_loss += loss.item()
        running_training_loss=running_training_loss/(count+1)
        print('running training loss = %2.5f' % running_training_loss)

        # Test
        model.eval()
        test_loss = 0.0
        total=0
        correct=0
        for count, (input, target) in enumerate(testloader):
            input, target = input.to(device), target.to(device)
            output = model(input) # forward pass
            loss = criterion(output, target) # loss
            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum()
        test_loss=test_loss/(count+1)
        test_accuracy_percentage=float(100.0*correct)/total
        print('test loss = %2.5f' % test_loss)
        print('test accuracy = %2.2f percent' % test_accuracy_percentage)

if __name__ == '__main__':
    main()
