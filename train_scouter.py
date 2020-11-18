from args_setting import *
from model.scouter.scouter_model import SlotModel
from model.model_tools import print_param
import torch
import time
import datetime
import tools.prepare_things as prt
from pathlib import Path
from engine_scouter import train_one_epoch, evaluate
from tools.calculate_tool import MetricLog
from loaders.base_loader import get_dataloader
import os
from tools.Adabelif import AdaBelief
import numpy as np


def main(args, selection=None):
    device = torch.device(args.device)
    loaders_train = get_dataloader(args, "train", selection=selection, mode="train")
    loaders_val = get_dataloader(args, "train", selection=selection, mode="val")
    model = SlotModel(args)
    model.to(device)
    print_param(model)
    params = [p for p in model.parameters() if p.requires_grad]
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    # optimizer = AdaBelief(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    print("Start training")
    start_time = time.time()
    log = MetricLog(args)
    record = log.record
    max_acc = 0

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(args, model, loaders_train, device, record, epoch, optimizer)
        evaluate(args, model, loaders_val, device, record, epoch)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / model_name]
            if record["val"]["acc"][epoch-1] > max_acc:
                print("get higher acc save current model")
                max_acc = record["val"]["acc"][epoch-1]
                for checkpoint_path in checkpoint_paths:
                    prt.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
        log.print_metric()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.fsl = False
    args.lr_drop = 40
    args.epochs = 60
    args.batch_size = 128
    args.use_slot = True
    args.slot_base_train = True
    args.drop_dim = False
    args.lr = 0.0001
    model_name = (f"{args.dataset}_" + f"{args.base_model}_" + f"{'use_slot_' if args.use_slot else 'no_slot_'}"
                  + f"{args.num_slot if args.use_slot else ''}" + 'checkpoint.pth')
    if args.random:
        selection = np.random.randint(0, args.num_classes, args.num_slot)
    else:
        selection = np.arange(0, args.num_classes, args.interval)
    print(selection)
    args.num_slot = len(selection)
    print("patterns num: ", args.num_slot)
    print("model name: ", model_name)
    main(args, selection=selection)

