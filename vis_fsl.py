import argparse
from args_setting import *
import matplotlib.cm as mpl_color_map
import copy
from model.FSL import FSLSimilarity, SimilarityLoss
from loaders.base_loader import get_dataloader
from PIL import Image
import numpy as np
import torch
import os

os.makedirs('vis/', exist_ok=True)
os.makedirs('vis/att', exist_ok=True)
os.makedirs('vis/all', exist_ok=True)


def test(args, model, image, record_name):
    image = image.to(device, dtype=torch.float32)
    b = image.size()[0]
    output, att = model(image)
    loss, acc, logits = criterion(output, att)

    for i in range(b):
        image_raw = Image.open(record_name[i]).convert('RGB').resize((args.img_size, args.img_size), resample=Image.BILINEAR)
        image_raw.save("vis/att/" + str(i) + "image.png")
        for id in range(args.num_slot):
            slot_image = np.array(Image.open(f'vis/att/{i}_slot_{id}.png'), dtype=np.uint8)
            heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, slot_image, 'jet')
            heatmap_on_image.save("vis/att/" + f'{i}_slot_mask_{id}.png')

        if i < args.n_shot*args.n_way:
            affine_name = "support"
            index = i
        else:
            affine_name = "query"
            index = i - args.n_shot*args.n_way

        # sum_slot = np.array(Image.open(f'vis/affine/affined_{affine_name}_{index}.png'), dtype=np.uint8)
        # heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, sum_slot, 'jet')
        # heatmap_on_image.save("vis/affine/" + f'colored_affined_{affine_name}_{index}.png')

        sum_slot = np.array(Image.open(f'vis/all/origin_{affine_name}_{index}.png'), dtype=np.uint8)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, sum_slot, 'jet')
        heatmap_on_image.save("vis/all/" + f'colored_origin_{affine_name}_{index}.png')


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
    model = FSLSimilarity(args)
    model_name = saved_name
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    sample_info_val = [args.val_episodes, args.n_way, args.n_shot, args.query]
    dataset_val = get_dataloader(args, "val", sample=sample_info_val, out_name=True, seed=seed)
    data = iter(dataset_val).next()

    imgs, labels, img_name = data
    print(imgs.size())
    test(args, model, imgs, img_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.random:
        selection = np.random.randint(0, args.num_classes, args.num_slot)
    else:
        selection = np.arange(0, args.num_classes, args.interval)
    args.num_slot = len(selection)
    print(args.num_slot)
    args.query = 1
    args.vis = True
    args.double = True
    args.fsl = True
    args.slot_base_train = False
    saved_name = (f"{args.dataset}_{args.base_model}_slot{args.num_slot}_" + 'fsl_checkpoint.pth')
    seed = None
    device = torch.device(args.device)
    criterion = SimilarityLoss(args).to(device)
    main()