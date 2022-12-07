#!/usr/bin/env python

import torch, datetime
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import argparse


## Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return x    # return x for visualization





def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def main():
    
    

    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--batch-size", type=int, default=16, metavar="N",
                        help="input batch size for training (default: 16)")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="input batch size for testing (default: 1000)")
    parser.add_argument("--epochs", type=int, default=10, metavar="N",
                        help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M",
                        help="SGD momentum (default: 0.5)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=1, metavar="S",
                        help="random seed (default: 1)")
    parser.add_argument("--val-size", type=int, default=5000, metavar="S",
                        help="Validation images (default: 5000)")
    parser.add_argument("--datadir", type=str, default="/data/mnist",
                        help="Path to download data to. Default is /data/mnist")
    parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--log-path", type=str, default="",
                        help="Path to save logs. Print to StdOut if log-path is not set")


    args = parser.parse_args()
  


    torch.manual_seed(args.seed)

    
    
    net=Net()
    print(net.parameters)
    #net=torchvision.models.convnext_large()
    
    if torch.cuda.is_available():
        device = 'cuda'
        net.cuda(torch.cuda.current_device());
    else:
        device = 'cpu'
  
    
    criterion = nn.CrossEntropyLoss()

    if device == 'cuda':
        criterion.cuda()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    

    
    # Get the dataset
    
    # Prepare training data
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    
    train_size = int(0.8 * len(dataset))
    val_size   = int(0.2 * len(dataset))
    print("t_size, v_size=",train_size,val_size)
    trainset, valset = random_split(dataset, [train_size, val_size])
    
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, num_workers=1, 
                                              pin_memory=True)
    
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=1,
                                            pin_memory=True)
    # Prepare test data
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=1,
                                            pin_memory=True)
    
    classes = dataset.classes 
    
    

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        # Train loop
        net.train()
        train_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0],data[1]
            if device == 'cuda':
                inputs=inputs.cuda()
                labels=labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_acc = accuracy(outputs,labels)
        train_loss = train_loss / len(trainloader.dataset)
    
        # Validation loop ( we won't backprop and optimize since this step is not training the model)
        net.eval()
        val_loss = 0.0
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0],data[1]
            if device == 'cuda':
                inputs=inputs.cuda()
                labels=labels.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() #* data[0].size(0)
    
        val_acc = accuracy(outputs,labels).item()
        val_loss = val_loss / len(valloader.dataset)
        print('epoch:',epoch)
        print('Validation-accuracy=%.3f | Validation-loss=%.3f'%(val_acc,val_loss))
        
    




if __name__ == "__main__":
    main()
