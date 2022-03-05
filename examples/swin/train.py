import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime
import time
import os
import numpy as np

from metrics import accuracy

def train(train_loader, model, criterion, optimizer, scheduler, scaler, epoch, device, writer_train, req_num_iterations):
    print_freq = 100
    
    # switch to train mode
    model.train()
    
    T = tqdm(enumerate(train_loader), desc=f'epoch {epoch}', position=0, leave=True)

    end = time.time()
    for i, (images, target) in T:
        # measure data loading time
        data_time = time.time() - end
        images = images.cuda(device)
        target = target.cuda(device)
        
        with torch.cuda.amp.autocast(enabled=True):       
            # compute output
            output = model(images).logits
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc = accuracy(output, target, topk=(1,))[0]
            scaler.scale(loss).backward()

        # compute gradient and do SGD step
        if (i + 1) % req_num_iterations == 0:
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
            scheduler.step_update(epoch * len(train_loader) + i)
             
        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()
        
        T.set_description(f"Epoch {epoch}, loss: {loss.item():.5f}, " +\
                          f"accuracy: {acc.item():.5f}, data_time: {data_time:.3f}, " +\
                          f"batch_time: {batch_time:.3f}, scale: {scaler._scale.item()}, lr: {optimizer.param_groups[0]['lr']}", refresh=False)


        if i % print_freq == 0:
            training_iter = epoch * len(train_loader) + i
            writer_train.add_scalar('loss', loss / images.shape[0], global_step = training_iter)
            writer_train.add_scalar('accuracy', acc / images.shape[0], global_step = training_iter)
            