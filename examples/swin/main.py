import torch
import torch.nn as nn
from transformers import ViTConfig, ViTForImageClassification
from transformers import SwinConfig, SwinForImageClassification
from timm.scheduler.cosine_lr import CosineLRScheduler
import hivemind
import argparse

from main_worker import main_worker
from dataloader import build_dataloader

class PointProver(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def main(args, model_name):
    configuration = SwinConfig()
    configuration.num_labels = 1000
    model = PointProver(SwinForImageClassification(configuration))

    batch_size = 32
    req_batch_size = 1024
    req_num_iterations = int(req_batch_size / batch_size)
    num_epochs = 300

    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(device)
    model.cuda(device)

    train_loader, val_loader = build_dataloader(batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), eps=1e-8, betas=(0.9, 0.999), lr=5e-4, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss().cuda(device)
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
    start_epoch = 0

    dht = hivemind.DHT(
        start=True,
        initial_peers=args.initial_peers,
    )
    print("To join the training, use initial_peers =", [str(addr) for addr in dht.get_visible_maddrs()])

    optimizer = hivemind.Optimizer(
        dht=dht,                  # use a DHT that is connected with other peers
        run_id='my_swin_run',    # unique identifier of this collaborative run
        batch_size_per_step=batch_size,   # each call to opt.step adds this many samples towards the next epoch
        target_batch_size=req_batch_size,  # after peers collectively process this many samples, average weights and begin the next epoch
        optimizer=optimizer,      # wrap the SGD optimizer defined above
        use_local_updates=True,   # perform optimizer steps with local gradients, average parameters in background
        matchmaking_time=3.0,     # when averaging parameters, gather peers in background for up to this many seconds
        averaging_timeout=10.0,   # give up on averaging if not successful in this many seconds
    )

    # saved_model_path = f"/external1/meshchaninov/MLEngineeringPractice/saved_models/swin_20220313/Epoch: 91, acc: 0.71957"
    # if saved_model_path:
    #     state = torch.load(saved_model_path, map_location=device)
    #     model.load_state_dict(state["state_dict"])
    #     optimizer.load_state_dict(state["optimizer"])
    #     scheduler.load_state_dict(state["scheduler"])
    #     scaler.load_state_dict(state["scaler"])
    #     start_epoch = state["epoch"]

    main_worker(num_epochs, model, optimizer, criterion, scheduler, scaler, model_name, train_loader, val_loader,
                req_num_iterations, device, start_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial_peers', default=None, help='To join the training, use --initial_peers=')
    args = parser.parse_args()
    main(args, "swin-hv")
