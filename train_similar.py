from args_setting import *
from model.model_tools import print_param
from model.FSL_similarity import FSLSimilarity
from model.FSL_similarity import SimilarityLoss
# from temp import SimilarityLoss
import torch
import time
import datetime
import tools.prepare_things as prt
from pathlib import Path
from engine_similarity import train_one_epoch, evaluate
from tools.calculate_tool import MetricLogSimilar
from tools.Adabelif import AdaBelief
from loaders.base_loader import make_loaders


def main(args):
    device = torch.device(args.device)
    loaders = make_loaders(args)
    criterien = SimilarityLoss(args).to(device)
    model = FSLSimilarity(args)

    model_name = "scouter_checkpoint.pth"
    model.to(device)
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])

    print_param(model)
    params = [p for p in model.parameters() if p.requires_grad]

    output_dir = Path(args.output_dir)
    # optimizer = torch.optim.AdamW(params, lr=args.lr)
    optimizer = AdaBelief(params, lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    print("Start training")
    start_time = time.time()
    log = MetricLogSimilar(args)
    record = log.record

    max_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(model, loaders["train"], device, record, epoch, optimizer, criterien)
        evaluate(model, loaders["val"], device, record, epoch, criterien)
        lr_scheduler.step()

        if args.output_dir:
            checkpoint_paths = [output_dir / ("similarity_checkpoint_ab_lambda0_fc_att" + str(epoch) + ".pth")]
            # if record["val"]["accm"][epoch-1] > max_acc1:
            #     print("get higher acc save current model")
            #     max_acc1 = record["val"]["accm"][epoch-1]
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