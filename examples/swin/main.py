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
from transformers import SwinModel, SwinConfig, SwinForImageClassification
from timm.scheduler.cosine_lr import CosineLRScheduler


from main_worker import main_worker
from dataloader import build_dataloader


def main(model_name):
    if model_name == "swin":
        configuration = SwinConfig()
        configuration.num_labels = 1000
        model = SwinForImageClassification(configuration)
    elif model_name == "vit":
        configuration = ViTConfig()
        configuration.num_labels = 1000
        model = ViTForImageClassification(configuration)
        
    batch_size = 128
    req_batch_size = 1024
    req_num_iterations = int(req_batch_size / batch_size)
    num_epochs = 300  
    
    train_loader, val_loader = build_dataloader(batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial=int(num_epochs * len(train_loader)),
            lr_min=5e-6,
            warmup_lr_init=5e-7,
            warmup_t=int(20 * len(train_loader)),
            cycle_limit=1,
            t_in_epochs=False,
        )
    scaler = torch.cuda.amp.GradScaler()
    
    main_worker(num_epochs, model, optimizer, criterion, scheduler, scaler, model_name, train_loader, val_loader, req_num_iterations)
    
    
if __name__ == '__main__':
    main("swin")
