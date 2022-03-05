import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm_notebook as tqdm
import datetime
import time
import os
import numpy as np

from metrics import accuracy


def validate(val_loader, model, criterion, epoch, device, writer_val):
    losses = []
    accs = []
    T = tqdm(enumerate(val_loader), desc=f'epoch {epoch}', position=0, leave=True)
    
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in T:
            data_time = time.time() - end
            images = images.cuda(device)
            target = target.cuda(device)

            # compute output
            output = model(images).logits
            loss = criterion(output, target)
            losses.append(loss.item() / images.shape[0])
            
            # measure accuracy and record loss
            acc = accuracy(output, target, topk=(1,))[0]
            accs.append(acc.item())
            
            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            T.set_description(f"Epoch: {epoch}, loss: {np.mean(losses):.5f}, accuracy: {np.mean(accs):.5f}, " +\
                                f"data_time: {data_time:.3f}, batch_time: {batch_time:.3f}", refresh=False)
            
    writer_val.add_scalar('loss', np.mean(losses), global_step = epoch)
    writer_val.add_scalar('accuracy', np.mean(accs), global_step = epoch)
    return np.mean(accs)