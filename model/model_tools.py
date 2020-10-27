import model.extractor as base_bone
import torch
import torch.nn as nn
from collections import OrderedDict


def fix_parameter(model, name_fix, mode="open"):
    """
    fix parameter for model training
    """
    for name, param in model.named_parameters():
        for i in range(len(name_fix)):
            if mode != "fix":
                if name_fix[i] not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    break
            else:
                if name_fix[i] in name:
                    param.requires_grad = False


def print_param(model):
    # show name of parameter could be trained in model
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


class Trans(nn.Module):
    def __init__(self):
        super(Trans, self).__init__()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return x


def load_backbone(args, drop=True):
    bone = base_bone.__dict__[args.base_model](num_classes=args.num_classes, drop_dim=drop)
    if args.use_slot:
        if args.use_pre:
            checkpoint = torch.load(f"saved_model/{args.dataset}_no_slot_checkpoint.pth")
            # new_state_dict = OrderedDict()
            # for k, v in checkpoint["model"].items():
            #     name = k[9:]  # remove `backbone.`
            #     new_state_dict[name] = v
            bone.load_state_dict(checkpoint["model"])
            print("load pre feature extractor parameter over")
        bone.avg_pool = Identical()
        bone.linear = Identical()
    return bone


