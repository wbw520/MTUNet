import os

import PIL.Image as Image
import numpy as np

__all__ = ['DatasetFolder']


class DatasetFolder(object):

    def __init__(self, root, set_name, split_type, transform, out_name=False):
        assert split_type in ['train', 'test', 'val']
        split_file = os.path.join("data/data_split", set_name, split_type + '.csv')
        assert os.path.isfile(split_file)
        with open(split_file, 'r') as f:
            split = [x.strip().split(',') for x in f.readlines()[1:] if x.strip() != '']
        data, ori_labels = [x[0] for x in split], [x[1] for x in split]
        label_key = sorted(np.unique(np.array(ori_labels)))
        label_map = dict(zip(label_key, range(len(label_key))))
        mapped_labels = [label_map[x] for x in ori_labels]

        self.split_type = split_type
        self.root = root + "/" + set_name
        self.transform = transform
        self.data = data
        self.labels = mapped_labels
        self.out_name = out_name
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        folder_name = self.data[index][:9]
        img_name = self.data[index]
        img_root = os.path.join(self.root, self.split_type, folder_name, img_name)
        assert os.path.isfile(img_root)
        img = Image.open(img_root).convert('RGB')
        label = self.labels[index]
        label = int(label)
        if self.transform:
            img = self.transform(img)
        if self.out_name:
            return img, label, img_name
        else:
            return img, label