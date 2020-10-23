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
os.makedirs('vis/support', exist_ok=True)
os.makedirs('vis/query', exist_ok=True)

def test(args, model, img, image, record_name):
    image = image.to(device, dtype=torch.float32)
    output, att = model(image)

    #For vis
    image_raw = img
    image_raw.save('vis/image.png')
    image_raw.save("vis/" + record_name + "image.png")
    model.train()

    for id in range(args.num_slot):
        image_raw = Image.open('vis/image.png').convert('RGB')
        slot_image = np.array(Image.open(f'vis/slot_{id}.png').resize(image_raw.size, resample=Image.BILINEAR), dtype=np.uint8)

        heatmap_only, heatmap_on_image = apply_colormap_on_image(image_raw, slot_image, 'jet')
        heatmap_on_image.save("vis/" + record_name + f'slot_mask_{id}.png')

    return output

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
    model_name = "similarity_checkpoint_ab_lambda0_fc_att0.pth"
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
    total_out = []
    for i in range(len(inputs_support)):
        total_out.append(iters(support_name[i], model, torch.unsqueeze(inputs_support[i], dim=0), "support/pic_" + str(cls[i].item()) + "_"))
    print("------------")
    for j in range(len(inputs_query)):
        total_out.append(iters(query_name[j], model, torch.unsqueeze(inputs_query[j], dim=0), "query/pic_" + str(cls[j].item()) + "_"))
    pp = torch.cat(total_out, dim=0)
    # print(labels_query.size())
    loss, acc = criterion(pp, 0, labels_query, 0, "val", model.classifier)
    # print(pp.size())

def iters(name, model, image, record_name):
    image_orl = Image.open(name[0])
    out = test(args, model, image_orl, image, record_name)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.batch_size = 1
    args.query = 1
    args.vis = True
    device = torch.device(args.device)
    criterion = SimilarityLoss(args).to(device)
    main()