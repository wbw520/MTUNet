import torch
from torch.utils.data import DataLoader, Dataset
import loaders.datasets as datasets
from args_setting import *


def get_li(data, cls, index):
    new_data = []
    new_cls = []
    for i in index:
        new_data.extend(data[i*600: (i+1)*600])
        new_cls.append(cls[i])
    return new_data, new_cls


def get_dataloader(args, split, shuffle=True, out_name=False, sample=None):
    # sample: iter, way, shot, query
    transform = datasets.make_transform(args, split)
    sets = datasets.DatasetFolder(args.data_root, args.dataset, split, transform, out_name=out_name)
    if sample is not None:
        sampler = datasets.CategoriesSampler(sets.labels, *sample)
        loader = torch.utils.data.DataLoader(sets, batch_sampler=sampler,
                                             num_workers=args.num_workers, pin_memory=True)
    else:
        loader = torch.utils.data.DataLoader(sets, batch_size=args.batch_size, shuffle=shuffle,
                                             num_workers=args.num_workers, pin_memory=True)
    return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    wbw =get_dataloader(args, "val")
    print(len(wbw))
    for i_batch, sample_batch in enumerate(wbw):
        # print(sample_batch["query"]["image"].size())
        # print(sample_batch["query"]["label"].size())
        # print(sample_batch["support"]["image"].size())
        # print(sample_batch["support"]["label"].size())
        # print(sample_batch["selected_cls"].size())
        # print(sample_batch["selected_cls"])
        print("-------------")