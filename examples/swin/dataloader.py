import torch
from torchvision import datasets, transforms, models

def build_dataloader(batch_size):
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
        batch_size = batch_size,
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
        batch_size = batch_size,
        shuffle = False,
        num_workers = 16,
        pin_memory=True)   
    
    return train_loader, val_loader