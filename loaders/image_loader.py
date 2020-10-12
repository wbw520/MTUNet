import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import numpy as np


class ImageLoader(Dataset):
    def __init__(self, args, data, class_index, mode, transform=None):
        super(ImageLoader, self).__init__()
        self.args = args
        self.image_root = args.data_root
        self.ratio = 0.9
        if not args.fsl:
            self.data = self.split(data)[mode]
            self.mode = "train"
        else:
            self.data = data
            self.mode = mode
        self.class_index = class_index
        self.transform = transform

    def deal_label(self, input):
        return self.class_index.index(input)

    def split(self, data):
        train, val = train_test_split(data, random_state=1, train_size=self.ratio)
        return {"train": train, "val": val}

    def __getitem__(self, index):
        file_name, label_ = self.data[index]
        image_path = os.path.join(self.image_root, self.mode, label_, file_name)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = self.deal_label(label_)
        label = torch.from_numpy(np.array(label))
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": label, "name": image_path}

    def __len__(self):
        return len(self.data)