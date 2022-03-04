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
from transformers import ViTModel, ViTConfig, ViTForImageClassification


class CustomSheduler:
    def __init__(self, optimizer, lr=0.1):
        self._optimizer = optimizer
        self.lr = lr
        self._optimizer.param_groups[0]['lr'] = lr

        
    def step(self, iteration):
        if iteration in [32_000, 48_000]:
            self._optimizer.param_groups[0]['lr'] /= 10


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def train(train_loader, model, criterion, optimizer, scheduler, epoch, device, writer_train):
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
        
        # compute output
        output = model(images).logits
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc = accuracy(output, target, topk=(1,))[0]

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
             
        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()
        
        T.set_description(f"Epoch {epoch}, loss: {loss.item() / images.shape[0]:.5f}, accuracy: {acc.item() / images.shape[0]:.5f}, data_time: {data_time:.3f}, batch_time: {batch_time:.3f}", refresh=False)

        if i % print_freq == 0:
            training_iter = epoch * len(train_loader) + i
            writer_train.add_scalar('loss', loss / images.shape[0], global_step = training_iter)
            writer_train.add_scalar('accuracy', acc / images.shape[0], global_step = training_iter)
        scheduler.step(epoch * len(train_loader) + i)
        
            
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
            accs.append(acc.item() / images.shape[0])
            
            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            T.set_description(f"Epoch: {epoch}, loss: {np.mean(losses):.5f}, accuracy: {np.mean(accs):.5f}, data_time: {data_time:.3f}, batch_time: {batch_time:.3f}", refresh=False)
            
    writer_val.add_scalar('loss', np.mean(losses), global_step = epoch)
    writer_val.add_scalar('accuracy', np.mean(accs), global_step = epoch)
    return np.mean(accs)


def main_worker(num_epochs):
    configuration = ViTConfig()
    configuration.num_labels = 1000
    model = ViTForImageClassification(configuration)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scheduler = CustomSheduler(opt, lr = 0.1)

    # Data loading code
    traindir = "/external2/imagenet/train"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = 48,
        shuffle = True,
        num_workers = 16,
        pin_memory = True)
    
    valdir = "/external2/imagenet/val"

    val_dataset = datasets.ImageFolder(
        valdir, 
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = 48,
        shuffle = False,
        num_workers = 16,
        pin_memory=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model = model.to(device)
    
    model_name = "vit"
    log_dir = f"/external1/meshchaninov/MLEngineeringPractice/tensorboard_logs/{model_name}_{datetime.datetime.now().strftime('%Y%m%d')}"
    save_dir = f"/external1/meshchaninov/MLEngineeringPractice/saved_models/{model_name}_{datetime.datetime.now().strftime('%Y%m%d')}"
    os.makedirs(log_dir, exist_ok = True)
    os.makedirs(save_dir, exist_ok = True)
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_val = SummaryWriter(os.path.join(log_dir, 'valid'))
    
    best_acc1 = -1
    for epoch in range(num_epochs):        
        train(train_loader, model, criterion, opt, scheduler, epoch, device, writer_train)
        acc1 = validate(val_loader, model, criterion, epoch, device, writer_val)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : opt.state_dict(),
            }

            torch.save(state, f"{save_dir}/Epoch: {epoch}, acc: {best_acc1:.5f}")
        

        
if __name__ == '__main__':
    main_worker(50)