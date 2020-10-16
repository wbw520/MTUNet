import argparse
from train import get_args_parser
from torch.utils.data import DataLoader, Dataset
import matplotlib.cm as mpl_color_map
import copy
from tqdm.auto import tqdm
import tools.calculate_tool as cal
from model.FSL_similarity import FSLSimilarity, SimilarityLoss
from loaders.base_loader import make_loaders
from PIL import Image
import numpy as np
import torch


@torch.no_grad()
def evaluate(model, data_loader, device, criterion):
    running_loss = 0.0
    running_acc_95 = []
    L = len(data_loader)
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs_query = torch.cat(list(sample_batch["query"]["image"].to(device, dtype=torch.float32)), dim=0)
        labels_query = sample_batch["query"]["label"].to(device, dtype=torch.int64)
        inputs_support = torch.cat(list(sample_batch["support"]["image"].to(device, dtype=torch.float32)), dim=0)
        labels_support = sample_batch["support"]["label"].to(device, dtype=torch.int64)
        total_input = torch.cat([inputs_support, inputs_query], dim=0)
        out, att_loss = model(total_input)
        loss, acc = criterion(out, labels_support, labels_query, att_loss)
        a = loss.item()
        running_loss += a
        running_acc_95.append(round(acc.item(), 4))

    print("loss: ", round(running_loss/L, 3))
    print("acc_95: ", round(cal.compute_confidence_interval(running_acc_95)[0], 4))
    print("interval: ", round(cal.compute_confidence_interval(running_acc_95)[1], 4))


def main():
    criterien = SimilarityLoss(args)
    model = FSLSimilarity(args)
    model_name = "similarity_checkpoint.pth"
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    dataset_test = make_loaders(args)["test"]
    evaluate(model, dataset_test, device, criterien)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    device = torch.device(args.device)
    main()