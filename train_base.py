from args_setting import *
from model.scouter.scouter_model import SlotModel
from model.model_tools import print_param
import torch
import time
import datetime
import tools.prepare_things as prt
from pathlib import Path
from engine_base import train_one_epoch
from tools.Adabelif import AdaBelief
from tools.calculate_tool import MetricLog
from loaders.base_loader import get_dataloader
import os


def main(args, selection=None):
    device = torch.device(args.device)
    loaders_train = get_dataloader(args, "train", selection=selection)
    model = SlotModel(args)
    model.to(device)
    print_param(model)
    params = [p for p in model.parameters() if p.requires_grad]

    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    optimizer = AdaBelief(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    print("Start training")
    start_time = time.time()
    log = MetricLog(args)
    record = log.record

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(args, model, loaders_train, device, record, epoch, optimizer)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / (f"{args.dataset}_" + f"{'use_slot_' if args.use_slot else 'no_slot_'}" + 'checkpoint.pth')]
            if epoch == args.epochs - 1:
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
    args.lr_drop = 30
    args.epochs = 80
    args.batch_size = 256
    args.use_slot = False
    args.drop_dim = True
    print("start base model training: ")
    main(args)
    args.use_slot = True
    args.drop_dim = False
    args.lr = 0.0001
    main(args, selection=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

