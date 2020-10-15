import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from loaders.transform_func import make_transform
from loaders.image_loader import ImageLoader
import os
from args_setting import *
import numpy as np
import random
import copy


def get_class(data):
    cls = {}
    for img in data:
        if img[1] not in cls:
            cls.update({img[1]: [img[0]]})
        else:
            cls[img[1]].append(img[0])
    return cls


def read_csv(name):
    data_all = pd.read_csv(name)
    wbw = data_all[["filename", "label"]]
    wbw = wbw.values
    return wbw, get_class(wbw)


class Data():
    def __init__(self, args):
        self.root = args.data_root

    def get_record(self):
        train_data, train_cls = read_csv(os.path.join(self.root, "train.csv"))
        val_data, val_cls = read_csv(os.path.join(self.root, "val.csv"))
        test_data, test_cls = read_csv(os.path.join(self.root, "test.csv"))
        return {"train": [train_data, train_cls], "val": [val_data, val_cls], "test": [test_data, test_cls]}


class FSLLoader(Dataset):
    def __init__(self, args, mode, way, n_classes, n_episodes, transform=None):
        super(FSLLoader, self).__init__()
        self.args = args
        self.n_episodes = n_episodes
        self.n_classes = n_classes
        self.n_way = way
        self.shot = args.n_shot
        self.query = args.query
        self.mode = mode
        self.transform = transform
        self.all_data = Data(args).get_record()
        self.data = self.all_data[mode][1]
        self.cls = list(self.data.keys())
        self.base_cls = list(self.all_data["train"][1].keys())
        self.for_all_class = False

    def __len__(self):
        return self.n_episodes

    def get_fsl_split(self):
        return torch.randperm(self.n_classes)[:self.n_way]

    def get_fsl_imgs(self, split):
        img_support = []
        img_query = []
        mini_batch_cls = []
        for st in split:
            current_cls = self.cls[st.item()]
            mini_batch_cls.append(current_cls)
            copy_imgs = copy.deepcopy(self.data[current_cls])
            random.shuffle(copy_imgs)
            for ss in copy_imgs[:self.shot]:
                img_support.append([ss, current_cls])
            for qq in copy_imgs[self.shot:self.shot+self.query]:
                img_query.append([qq, current_cls])
        return img_support, img_query, mini_batch_cls

    def over_loader(self, loader):
        imgs = []
        labels = []
        name = []
        for i_batch, sample in enumerate(loader):
            imgs.append(torch.unsqueeze(sample["image"], dim=0))
            labels.append(torch.unsqueeze(sample["label"], dim=0))
            name.append(sample["name"])
        return {"image": torch.cat(imgs, dim=0), "label": torch.cat(labels, dim=0), "name": name}

    def __getitem__(self, index):
        selected_cls = self.get_fsl_split()
        support, query, mini_batch_cls = self.get_fsl_imgs(selected_cls)
        if self.for_all_class:
            if self.mode != "train":
                all_cls = self.base_cls + mini_batch_cls
            else:
                all_cls = self.base_cls
        else:
            all_cls = mini_batch_cls
        support_loader = ImageLoader(self.args, support, all_cls, self.mode, transform=make_transform(self.args, self.mode))
        query_loader = ImageLoader(self.args, query, all_cls, self.mode, transform=make_transform(self.args, self.mode))
        mini_batch_support = self.over_loader(support_loader)
        mini_batch_query = self.over_loader(query_loader)
        return {"selected_cls": selected_cls, "support": mini_batch_support, "query": mini_batch_query}


def make_loaders(args):
    if not args.fsl:
        all_data = Data(args).get_record()
        train_data = all_data["train"][0]
        cls = list(all_data["train"][1].keys())
        dataset_train = ImageLoader(args, train_data, cls, "train", transform=make_transform(args, "train"))
        dataset_val = ImageLoader(args, train_data, cls, "val", transform=make_transform(args, "val"))
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)
        print("load normal data over")
        return {"train": data_loader_train, "val": data_loader_val}
    else:
        dataset_train = FSLLoader(args, "train", args.n_way_train, args.train_classes, args.train_episodes)
        dataset_val = FSLLoader(args, "val", args.n_way, args.val_classes, args.val_episodes)
        dataset_test = FSLLoader(args, "test", args.n_way, args.test_classes, args.test_episodes)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, num_workers=args.num_workers)
        print("load FSL data over")
        return {"train": data_loader_train, "val": data_loader_val, "test": data_loader_test}


def make_loader_simple(args):
    all_data = Data(args).get_record()
    train_data = all_data["train"][0]
    cls = list(all_data["train"][1].keys())
    dataset_train = ImageLoader(args, train_data, cls, "train", transform=make_transform(args, "train"))
    dataset_val = FSLLoader(args, "val", args.n_way, args.val_classes, args.val_episodes)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, 256, drop_last=True)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)
    print("load simple few over")
    return {"train": data_loader_train, "val": data_loader_val}


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     wbw = make_loaders(args)["train"]
#     print(len(wbw))
#     for i_batch, sample_batch in enumerate(wbw):
#         # print(sample_batch["query"]["image"].size())
#         # print(sample_batch["query"]["label"].size())
#         # print(sample_batch["support"]["image"].size())
#         # print(sample_batch["support"]["label"].size())
#         # print(sample_batch["selected_cls"].size())
#         # print(sample_batch["selected_cls"])
#         print("-------------")