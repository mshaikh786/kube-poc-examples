#!/usr/bin/env python
# coding: utf-8

# ## Boilerplate code

# In[1]:


import time, gc, datetime

# Timing utilities
start_time = None

def start_timer():
    global start_time
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.synchronize()
    start_time = time.time()

def end_timer_and_print(local_msg):
    torch.cuda.synchronize()
    end_time = time.time()
    print("\n" + local_msg)
    print("Total execution time = {:.3f} sec".format(end_time - start_time))
    print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))




import torch, datetime, os, argparse, re

# Business as usual
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.cuda import amp

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np



# In[3]:


torch.manual_seed(43)
cudnn.deterministic = True
cudnn.benchmark = False



# import and instantiate tensorboard for monitoring model performance
from tensorboardX  import SummaryWriter


# ### Additional package 
# Required for DDP implementation


import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Learning rate scheduler for progressively modifying LR w.r.t epochs to improve training
from torch.optim.lr_scheduler import StepLR


# Setting resources and variables for training in a Jupyter notebook.
# In a python script version of the code, this section should be parsed in as arguments.

# ## Miscellaneous utility funtions


def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return torch.sum(preds == labels).item()


# ## DataLoader
# Add a data management section to load and transform data.
# Here we manage not only the data location but also how it is loaded into memory.
# 
# ***NOTE***: `shuffle=True` when set in `trainSampler` makes the Dataloading buggy only if PyTorch version is > 1.12. The `if` condition takes care of it.

# In[7]:


def dataloader(gpu,world_size,batch_size,num_workers):
    
    trainSampler_shuffle=True 
#    version=float(re.findall(r'\d+\.\d+', torch.__version__)[0])
#   if version > 1.12:
#        print('Setting shuffle=False in trainSampler')
#        trainSampler_shuffle=False 
    trainSampler_shuffle=False
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
# Prepare training data
    train_transform = transforms.Compose([ 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),normalize ])

    val_transform = transforms.Compose([ 
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),normalize ])
    

    
    datadir=os.environ['DATA_DIR']
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(datadir,'train'),
                                                transform=train_transform)
    trainSampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                               num_replicas=world_size,
                                                               rank=gpu,
                                                               shuffle=trainSampler_shuffle,
                                                               drop_last=True)
    trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size,
                                          shuffle=False, 
                                          num_workers=num_workers,
                                          pin_memory=True,
                                          sampler=trainSampler)
                                         

    valset = torchvision.datasets.ImageFolder(root=os.path.join(datadir,'val'),
                                              transform=val_transform)
    valSampler = torch.utils.data.distributed.DistributedSampler(valset,
                                                                  num_replicas=world_size,
                                                                  rank=gpu,
                                                                  shuffle=False)
    valloader = torch.utils.data.DataLoader(valset, 
                                             batch_size=batch_size,
                                             shuffle=False, 
                                             num_workers=num_workers,
                                             pin_memory=True,
                                             sampler=valSampler,
                                             drop_last=True)
    return trainloader,valloader


# ## Choose a Neural Network architecture

# Pre-training
net=torchvision.models.resnet50(weights=None,num_classes=200)
# Transfer learning
#net=torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)


# ## Training
# Some additions and modifications are required to your training section. E.g.
# - Define a function for setting up multiple GPU context (using awareness of the environment)
#     - Here you can select the backend or the communication library to move data between memory of GPUs
# - Define a function and add the training steps in it
#     - Wrap model in DistributedDataParallel class
#     - The model, loss function and optimizer needs to be offloaded to each device using the corresponding gpu_id
#     - Figure out which tasks will be done exclusively master process (gpu_id==0)
#         - e.g. printing, writing tensorboard logs, saving and loading checkpoints etc
#     - Optionally, collect training accurracy and loss metrics on GPU 0 so it can write to tensorboard logs
# - Define a function that setups up the training environment and then calls the training
# 
# 
def setup():

    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    dist_url = "env://" # default

    # only works with torch.distributed.launch // torch.run
    rank = int(os.environ.get("RANK"))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    print (f"L-{local_rank} : W-{world_size} , R-{rank} \n")
    dist.init_process_group(
            backend="nccl",
            init_method=dist_url,
            world_size=world_size,
            rank=rank)

def cleanup():
    dist.destroy_process_group()

def train (net,args):
    rank = int(os.environ.get("RANK", 1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if int(os.environ.get('LOCAL_WORLD_SIZE')) == 1:
        gpu_id = int(os.environ.get("RANK", 0))
    else:
        gpu_id = int(os.environ.get("LOCAL_RANK", 0))
    
    # this will make all .cuda() calls work properly

    torch.cuda.set_device(gpu_id)
    # synchronizes all the threads to reach this point before moving on
    #dist.barrier()
    # Instantiate Tensorboard writer on process handler for GPU 0
    if rank == 0:
        writer = SummaryWriter("logs/%s" %('{}_{:%Y%m%d%H%M}'.format('experiment', datetime.datetime.now())))
    
    # Enable AMP
    scaler = amp.GradScaler()
    net.cuda()
    
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(net.parameters(), 
                          lr=args.lr, 
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    
    # [Optional]: Set LR scheduler
    scheduler =  StepLR(optimizer,step_size=30, gamma=0.1)
    
    trainloader, valloader = dataloader(gpu_id,world_size,
                                        args.batch_size,
                                        args.num_workers)
    # Wrap model as DDP
    net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[gpu_id])
    start_timer()
    print('Starting training on GPU %d of %d -- ' %(rank,world_size))
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        train_loss = 0.0
        train_acc  = 0
        trainloader.sampler.set_epoch(epoch)
        net.train()
        print(f'{rank}: Entering training loop for epoch {epoch}')
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].cuda(), data[1].cuda()
        
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=True,
                                       dtype=torch.float32):
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            #loss.backward()
            #optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_acc  += accuracy(outputs,labels)           
            
        valloader.sampler.set_epoch(epoch)
        del data
        val_loss = 0.0
        val_acc  = 0
        net.eval()
        print(f'{rank}: Entering validation for epoch {epoch}')
        for i, data in enumerate(valloader):
            inputs, labels = data[0].cuda(), data[1].cuda()
            with torch.no_grad():
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() 
            val_acc  += accuracy(outputs,labels)
            
        print(f'{rank}: Collecting metric for epoch {epoch}')
        # Gather accuracy metric from all training units on GPU 0  
        # to calculate an average over the size training dataset 
        train_loss = torch.tensor(train_loss).cuda()
        dist.reduce(train_loss,0,dist.ReduceOp.SUM)
        train_acc = torch.tensor(train_acc).cuda()
        dist.reduce(train_acc,0,dist.ReduceOp.SUM)
        
        val_loss = torch.tensor(val_loss).cuda()
        dist.reduce(val_loss,0,dist.ReduceOp.SUM)
        val_acc = torch.tensor(val_acc).cuda()  
        dist.reduce(val_acc,0,dist.ReduceOp.SUM)

        # Print from GPU 0
        if rank == 0:
            print(f'{rank}: Writing metric for epoch {epoch}')
            train_loss = train_loss.item() / len(trainloader.dataset.targets)
            train_acc  = 100 * (train_acc.item() / len(trainloader.dataset.targets))
            
            val_loss   = val_loss.item() / len(valloader.dataset.targets)
            val_acc    = 100 * (val_acc.item() / len(valloader.dataset.targets))

            print(f'[{epoch + 1}] :Loss (train, val):{train_loss:.3f}, {val_loss:.3f}| Accuracy (train,val): {train_acc:.3f}, {val_acc:.3f}')
            writer.add_scalar("Loss/train", train_loss , epoch)
            writer.add_scalar("Accuracy/train", train_acc , epoch)
            writer.add_scalar("Loss/val", val_loss , epoch)
            writer.add_scalar("Accuracy/val", val_acc , epoch)
            writer.flush
        
        # Save checkpoint every 10th epoch
        if (epoch+1) % 10 == 0:
            if rank == 0:
                PATH='./model_chkpt_ep%d.pth' %(epoch)
                torch.save(net.state_dict(), PATH)
                
        scheduler.step()
        
    if rank == 0:
        end_timer_and_print('Finished Training')
        writer.close()


def main(net,args):
    setup()
    for name, value in os.environ.items():
        print("{0}: {1}".format(name, value))
    train(net,args)
    return True

# Uncomment when using as python script
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", default=10,
                        help="number of dataloaders", type=int)
    parser.add_argument("--batch-size", default=256,
                        help="mini batch size per GPU", type=int)
    parser.add_argument("--epochs", default=5,
                        help="total epochs", type=int)
    parser.add_argument("--lr", default=0.1,
                        help="Learning rate",type=float)
    parser.add_argument("--momentum", default=0.9,
                        help="Momentum", type=float)
    parser.add_argument("--weight-decay", default=1e-4,
                        help="Momentum", type=float)
    parser.add_argument("--print-interval", default=100,
                        help="Momentum", type=int)
    args = parser.parse_args()
 
    main(net,args)


