from args_setting import *
from torch.utils.data import DataLoader, Dataset
import matplotlib.cm as mpl_color_map
import copy
from tqdm.auto import tqdm
import tools.calculate_tool as cal
from model.FSL import FSLSimilarity, SimilarityLoss
from loaders.base_loader import get_dataloader
from PIL import Image
import numpy as np
import torch


@torch.no_grad()
def evaluate(model, data_loader, device, criterion):
    model.eval()
    criterion.eval()
    print("start test: ")
    running_loss = 0.0
    running_att_loss = 0.0
    running_acc_95 = []
    L = len(data_loader)
    for i, (inputs, target) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device, dtype=torch.float32)
        out, att_loss = model(inputs)
        loss, acc, logits = criterion(out, att_loss)
        a = loss.item()
        running_loss += a
        running_att_loss += att_loss.item()
        running_acc_95.append(round(acc.item(), 4))

    record.append(round(cal.compute_confidence_interval(running_acc_95)[0], 4))
    print("loss: ", round(running_loss/L, 3))
    print("acc_95: ", round(cal.compute_confidence_interval(running_acc_95)[0], 4))
    print("interval: ", round(cal.compute_confidence_interval(running_acc_95)[1], 4))


def main(name):
    criterien = SimilarityLoss(args)
    model = FSLSimilarity(args)
    model_name = name
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    sample_info_val = [args.test_episodes, args.n_way, args.n_shot, args.query]
    loaders_test = get_dataloader(args, "test", sample=sample_info_val)
    evaluate(model, loaders_test, device, criterien)


if __name__ == '__main__':
    record = []
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    args.slot_base_train = False
    if args.random:
        selection = np.random.randint(0, args.num_classes, args.num_slot)
    else:
        selection = np.arange(0, args.num_classes, args.interval)
    print(selection)
    args.num_slot = len(selection)
    model_name = (f"{args.dataset}_{args.base_model}_slot{args.num_slot}_" + 'fsl_checkpoint.pth')
    main(model_name)
    # for i in range(0, 1):
    #     pp = "selction" + str(i+1) + "_cifar100_resnet18_slot7_fsl_checkpoint.pth"
    #     main(pp)
    #     print(record)