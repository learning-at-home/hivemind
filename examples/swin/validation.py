import torch
from tqdm import tqdm
import time

from metrics import accuracy


def validate(val_loader, model, criterion, epoch, device, writer_val):
    loss_epoch = 0.0
    acc_epoch = 0.0
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
            loss_epoch += loss.item()

            # measure accuracy and record loss
            acc = accuracy(output, target)
            acc_epoch += acc.item()

            # measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            T.set_description(f"Epoch: {epoch}, loss: {loss_epoch / (i + 1):.5f}, accuracy: {acc_epoch / (i + 1):.5f}, " + \
                              f"data_time: {data_time:.3f}, batch_time: {batch_time:.3f}", refresh=False)

    writer_val.add_scalar('test_loss', loss_epoch / len(val_loader), global_step=epoch)
    writer_val.add_scalar('test_accuracy', acc_epoch / len(val_loader), global_step=epoch)
    return acc_epoch / len(val_loader)
