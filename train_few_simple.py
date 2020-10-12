from args_setting import *
from model.scouter.scouter_model import SlotModel
from model.model_tools import print_param
from model.FSL import FSLPre, FSL, FSLLoss
import torch
import time
import datetime
import tools.prepare_things as prt
from pathlib import Path
import model.extractor as base_bone
from engine_few_simple import train_one_epoch, evaluate
from tools.calculate_tool import MetricLog
from loaders.base_loader import make_loader_simple


def main(args):
    device = torch.device(args.device)
    loaders = make_loader_simple(args)
    criterien = None
    model = base_bone.__dict__[args.base_model](num_classes=args.num_classes, drop_dim=True, extract=True)
    model.to(device)
    mark = "acc"
    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print('number of params:', n_parameters)
    print_param(model)
    params = [p for p in model.parameters() if p.requires_grad]

    output_dir = Path(args.output_dir)
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    # optimizer = torch.optim.SGD(params, momentum=0.9, lr=args.lr,
    #                             weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    print("Start training")
    start_time = time.time()
    log = MetricLog(args)
    record = log.record

    max_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(args, model, loaders["train"], device, record, epoch, optimizer, criterien)
        evaluate(args, model, loaders["val"], device, record, epoch, criterien)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / 'simple_few_checkpoint.pth']
            if record["val"][mark][epoch-1] > max_acc1:
                print("get higher acc save current model")
                max_acc1 = record["val"][mark][epoch]
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
    main(args)