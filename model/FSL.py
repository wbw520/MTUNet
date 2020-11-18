import torch
import model.extractor as base_bone
from model.model_tools import Identical, fix_parameter
from model.scouter.scouter_attention import ScouterAttention
from model.scouter.position_encode import build_position_encoding
import torch.nn as nn
from PIL import Image
import torch.nn.functional as F
import imgaug.augmenters as iaa
from torch.autograd import Variable
import numpy as np


def load_base(args):
    bone = base_bone.__dict__[args.base_model](num_classes=args.num_classes, drop_dim=args.drop_dim, extract=False)
    model_name = f"{args.dataset}_{args.base_model}_no_slot_checkpoint.pth"
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    bone.load_state_dict(checkpoint["model"], strict=True)
    print("load pre-model " + model_name + " ready")
    bone.avg_pool = Identical()
    bone.linear = Identical()
    return bone


class FSLSimilarity(nn.Module):
    def __init__(self, args):
        super(FSLSimilarity, self).__init__()
        self.args = args
        self.backbone = load_base(args)
        fix_parameter(self.backbone, [""], mode="fix")
        # fix_parameter(self.extractor, ["layer4", "layer3"], mode="open")
        self.channel = args.channel
        self.slots_per_class = args.slots_per_class
        self.conv1x1 = nn.Conv2d(self.channel, args.hidden_dim, kernel_size=(1, 1), stride=(1, 1))
        self.slot = ScouterAttention(args, self.slots_per_class, args.hidden_dim, vis=args.vis,
                                     vis_id=args.vis_id, loss_status=args.loss_status, power=args.power, to_k_layer=args.to_k_layer)
        fix_parameter(self.conv1x1, [""], mode="fix")
        fix_parameter(self.slot, [""], mode="fix")
        self.position_emb = build_position_encoding('sine', hidden_dim=args.hidden_dim)
        self.lambda_value = float(args.lambda_value)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                                        nn.LayerNorm(args.channel*2),
                                        nn.Dropout(0.5),
                                        nn.Linear(args.channel*2, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048, 1),
                                        nn.Sigmoid(),
        )
        self.use_threshold = False
        self.u_vis =args.vis

    def forward(self, x):
        x_pe, x, x_raw = self.feature_deal(x)
        x_raw = torch.relu(x_raw)
        x, attn_loss, attn = self.slot(x_pe, x)
        attn_support = attn[:self.args.n_way*self.args.n_shot, :, :]
        attn_query = attn[self.args.n_way*self.args.n_shot:, :, :]
        x_raw_support = x_raw[:self.args.n_way*self.args.n_shot]
        x_raw_query = x_raw[self.args.n_way*self.args.n_shot:]
        size = x_raw_support.size()[-1]
        dim = x_raw_support.size()[1]

        attn_support = attn_support.mean(1, keepdim=True).reshape(self.args.n_way*self.args.n_shot, 1, size, size)
        attn_query = attn_query.mean(1, keepdim=True).reshape(self.args.n_way*self.args.query, 1, size, size)

        if self.use_threshold:
            attn_support = self.threshold(attn_support)
            attn_query = self.threshold(attn_query)

        if self.u_vis:
            self.vis(attn_support, "origin_support", self.u_vis)
            # attn_support = self.affine(attn_support)
            # self.vis(attn_support, "affined_support", self.u_vis)
            self.vis(attn_query, "origin_query", self.u_vis)
            # attn_query = self.affine(attn_query)
            # self.vis(attn_query, "affined_query", self.u_vis)

        weighted_support = torch.mean(attn_support*(x_raw_support.reshape(self.args.n_way*self.args.n_shot, dim, size, size)), dim=(2, 3))
        weighted_query = torch.mean(attn_query*(x_raw_query.reshape(self.args.n_way*self.args.query, dim, size, size)), dim=(2, 3))

        if self.args.n_shot == 1:
            weighted_support = weighted_support.unsqueeze(0).expand(self.args.n_way*self.args.query, -1, -1)
        else:
            weighted_support = weighted_support.unsqueeze(0).expand(self.args.n_way*self.args.query, -1, -1).reshape(self.args.n_way*self.args.query, self.args.n_way, self.args.n_shot, -1).mean(-2)
        weighted_query = weighted_query.unsqueeze(1).expand(-1, self.args.n_way, -1)
        input_fc = torch.cat([weighted_support, weighted_query], dim=-1)
        out_fc = self.classifier(input_fc).squeeze(-1)
        return out_fc, attn_loss

    def threshold(self, data):
        mean_value = data.mean()
        data[data < mean_value] = 0.
        return data

    def feature_deal(self, x):
        x_raw = self.backbone(x)
        x = self.conv1x1(x_raw)
        pe = self.position_emb(x)
        x_pe = x + pe
        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0, 2, 1))
        x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
        return x_pe, x, x_raw

    def vis(self, att_vises, name, use_vis):
        if not use_vis:
            return
        b = att_vises.size(0)
        for i in range(b):
            att_vis = att_vises[i]
            att_vis = ((att_vis - att_vis.min()) / (att_vis.max()-att_vis.min()) * 255.)
            att_vis = (att_vis.squeeze(0).cpu().detach().numpy()).astype(np.uint8)
            image = Image.fromarray(att_vis, mode='L').resize((self.args.img_size, self.args.img_size), resample=Image.BILINEAR)
            image.save(f'vis/all/{name}_{i}.png')

    def affine(self, data):
        data = data.permute((0, 2, 3, 1)).cpu().detach().numpy()
        seq = iaa.Sequential([iaa.Affine(translate_percent={"x": (-0.6, 0.6), "y": (-0.6, 0.6)}, mode="wrap")])
        im = seq(images=np.array(data))
        im = torch.from_numpy(im)
        im = im.permute((0, 3, 1, 2)).cuda()
        return im


class SimilarityLoss(nn.Module):
    def __init__(self, args):
        super(SimilarityLoss, self).__init__()
        self.args = args
        self.BCEloss = nn.BCELoss()

    def get_slots(self, input, max):
        return torch.gather(input, 3, max)

    def forward(self, out_fc, att_loss):
        if self.args.vis:
            print("matching matrix:  ")
            print(np.round(out_fc.cpu().detach().numpy(), decimals=2))
        labels_query = Variable(torch.arange(0, self.args.n_way).view(self.args.n_way, 1).expand(self.args.n_way, self.args.query).long().cuda(), requires_grad=False).reshape(-1)
        labels_query_onehot = torch.zeros(labels_query.size()+(5,), dtype=labels_query.dtype).to(labels_query.device)
        labels_query_onehot.scatter_(-1, labels_query.unsqueeze(-1), 1)
        BCELoss = self.BCEloss(out_fc, labels_query_onehot.float())
        loss = BCELoss + float(self.args.lambda_value) * att_loss
        logits = F.log_softmax(out_fc, dim=-1)
        _, y_hat = logits.max(1)
        acc = torch.eq(y_hat, labels_query).float().mean()
        return loss, acc, logits