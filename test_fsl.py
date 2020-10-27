import argparse
from train import get_args_parser
from torch.utils.data import DataLoader, Dataset
import matplotlib.cm as mpl_color_map
import copy
from model.FSL_similarity import FSLSimilarity, SimilarityLoss
from loaders.base_loader import make_loaders
from PIL import Image
import numpy as np
import torch
import os

os.makedirs('vis/', exist_ok=True)
# os.makedirs('vis/support', exist_ok=True)
# os.makedirs('vis/query', exist_ok=True)


def test(args, model, image, record_name):
    image = image.to(device, dtype=torch.float32)
    b = image.size()[0]
    output, att = model(image)

    for i in range(b):
        image_raw = Image.open(record_name[i][0])
        image_raw.save('vis/image.png')
        image_raw.save("vis/" + str(i) + "image.png")
        for id in range(args.num_slot):
            image_raw = Image.open('vis/image.png').convert('RGB')
            slot_image = np.array(Image.open(f'vis/{i}_slot_{id}.png').resize(image_raw.size, resample=Image.BILINEAR), dtype=np.uint8)

            heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, slot_image, 'jet')
            heatmap_on_image.save("vis/" + f'{i}_slot_mask_{id}.png')


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
    model_name = "scouter_FSL_noslot24.pth"
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    dataset_val = make_loaders(args)["val"]
    data = iter(dataset_val).next()
    inputs_query = data["query"]["image"][0]
    query_name = data["query"]["name"]
    labels_query = data["query"]["label"].to(device, dtype=torch.int64)
    inputs_support = data["support"]["image"][0]
    support_name = data["support"]["name"]
    cls = data["selected_cls"][0]
    print(cls)
    total_input = torch.cat([inputs_support, inputs_query], dim=0)
    record_name = support_name + query_name
    print(record_name)
    test(args, model, total_input, record_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.batch_size = 1
    args.num_slot = 7
    args.query = 1
    args.vis = True
    args.fsl = True
    device = torch.device(args.device)
    criterion = SimilarityLoss(args).to(device)
    main()