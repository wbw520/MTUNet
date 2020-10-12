import argparse
from train import get_args_parser
from loaders.base_loader import Data, ImageLoader, make_transform
from torch.utils.data import DataLoader, Dataset
import matplotlib.cm as mpl_color_map
import copy
from model.scouter.scouter_model import SlotModel
from PIL import Image
import numpy as np
import torch


def test(args, model, img, image, label, vis_id):
    image = image.to(device, dtype=torch.float32)
    output = model(image)
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    print(output[0])
    print(pred[0])

    #For vis
    image_raw = img
    image_raw.save('vis/image.png')
    print(torch.argmax(output[vis_id]).item())
    model.train()

    for id in range(args.num_classes):
        image_raw = Image.open('vis/image.png').convert('RGB')
        slot_image = np.array(Image.open(f'vis/slot_{id}.png').resize(image_raw.size, resample=Image.BILINEAR), dtype=np.uint8)

        heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, slot_image, 'jet')
        heatmap_on_image.save(f'vis/slot_mask_{id}.png')

    if args.cal_area_size:
        slot_image = np.array(Image.open(f'vis/slot_{str(label) if args.loss_status>0 else str(label+1)}.png'), dtype=np.uint8)
        slot_image_size = slot_image.shape
        attention_ratio = float(slot_image.sum()) / float(slot_image_size[0]*slot_image_size[1]*255)
        print(f"attention_ratio: {attention_ratio}")


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def main():
    model = SlotModel(args)
    all_data = Data(args).get_record()
    train_data = all_data["val"][0]
    cls = list(all_data["val"][1].keys())
    cls2 = list(all_data["train"][1].keys())
    print(cls2)
    dataset_val = ImageLoader(args, train_data, cls, "val", transform=make_transform(args, "val"))
    data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    data = iter(data_loader_val).next()
    image = data["image"]
    label = data["label"]
    name = data["name"]
    print(name)
    image_orl = Image.open(name[0])
    model_name = f"{args.dataset}_" + 'use_slot_no_fsl_checkpoint.pth'
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    test(args, model, image_orl, image, label, vis_id=args.vis_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.batch_size = 1
    device = torch.device(args.device)
    main()