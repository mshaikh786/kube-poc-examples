import torch
from torchvision import datasets,models

def get_model():
    model = models.alexnet()
    return model

def compile_and_train(model,epochs):
    print('model definition')
    print(model.get_parameter)
    return model


def split_dataset(data_dir):
    train_dataset = datasets.MNIST(root=data_dir,
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)
    train_sampler = torch.utils.data.RandomSampler(data_source=train_dataset)
    train_loader  = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=4,
                                               sampler=train_sampler)
    
    
    test_dataset = datasets.MNIST(root=data_dir,
                                   train=False,
                                   transform=transforms.ToTensor(),
                                   download=True)
    test_sampler = torch.utils.data.RandomSampler(data_source=test_dataset)
    test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               sampler=test_sampler)
    
    return train_loader,test_loader
