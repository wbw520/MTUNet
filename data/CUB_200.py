import argparse
from sklearn.model_selection import train_test_split
import os
import shutil
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to the data', default="/home/wbw/PAN/CUB200")
parser.add_argument('--split', type=str, help='path to the split folder', default="/home/wbw/PAN/FSL_data")
args = parser.parse_args()


def read_txt(name):
    with open(name, 'r') as f:
        split = [x.strip().split(' ')[1] for x in f.readlines() if x.strip() != '']
    return split


def make_csv(data, name):
    f_val = open(name + ".csv", "w", encoding="utf-8")
    csv_writer = csv.writer(f_val)
    csv_writer.writerow(["filename", "label"])
    for i in range(len(data)):
        csv_writer.writerow(data[i])
    f_val.close()


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def move_data(split, phase, save_folder_root):
    image_name = []
    for cls in split:
        shutil.copytree(os.path.join(image_root, cls), os.path.join(save_folder_root, cls))
        imgs = get_name(os.path.join(image_root, cls), mode_folder=False)
        for name in imgs:
            image_name.append([cls + "/" + name, cls])
    make_csv(image_name, phase)


if __name__ == '__main__':
    data_root = args.data
    save_root = args.split
    image_root = data_root + "/CUB_200_2011/CUB_200_2011/images"
    category_txt = data_root + "/CUB_200_2011/CUB_200_2011/classes.txt"
    all_class = read_txt(category_txt)
    train_class, rest = train_test_split(all_class, random_state=1, train_size=0.5)
    val_class, test_class = train_test_split(rest, random_state=1, test_size=0.5)
    save_train_root = os.path.join(save_root, "CUB200", "train")
    save_val_root = os.path.join(save_root, "CUB200", "val")
    save_test_root = os.path.join(save_root, "CUB200", "test")
    os.mkdir(save_train_root)
    os.mkdir(save_val_root)
    os.mkdir(save_test_root)
    move_data(train_class, "data_split/CUB200/train", save_train_root)
    move_data(val_class, "data_split/CUB200/val", save_val_root)
    move_data(test_class, "data_split/CUB200/test", save_test_root)
