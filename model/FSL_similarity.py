import torch
import model.extractor as base_bone
from model.model_tools import Identical, fix_parameter
from model.scouter.scouter_attention import ScouterAttention
from model.scouter.position_encode import build_position_encoding
import torch.nn as nn
import torch.nn.functional as F


def load_base(args):
    bone = base_bone.__dict__[args.base_model](num_classes=args.num_classes, drop_dim=False, extract=False)
    checkpoint = torch.load(f"{args.output_dir}/" + "simple_few_checkpoint.pth", map_location=args.device)
    bone.load_state_dict(checkpoint["model"])
    bone.avg_pool = Identical()
    bone.linear = Identical()
    return bone


class FSLSimilarity(nn.Module):
    def __init__(self, args):
        super(FSLSimilarity, self).__init__()
        self.args = args
        self.extractor = load_base(args)
        fix_parameter(self.extractor, [""], mode="fix")
        fix_parameter(self.extractor, ["layer4", "layer3"], mode="open")
        self.channel = args.channel
        self.slots_per_class = args.slots_per_class
        self.conv1x1 = nn.Conv2d(self.channel, args.hidden_dim, kernel_size=(1, 1), stride=(1, 1))
        self.slot = ScouterAttention(args, args.n_way, self.slots_per_class, args.hidden_dim, vis=args.vis,
                                     vis_id=args.vis_id, loss_status=args.loss_status, power=args.power, to_k_layer=args.to_k_layer)
        self.position_emb = build_position_encoding('sine', hidden_dim=args.hidden_dim)
        self.lambda_value = float(args.lambda_value)

    def forward(self, x):
        x_pe, x = self.feature_deal(x)
        x, attn_loss = self.slot(x_pe, x)
        return x, attn_loss

    def feature_deal(self, x):
        x = self.extractor(x)
        x = self.conv1x1(x)
        pe = self.position_emb(x)
        x_pe = x + pe
        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0, 2, 1))
        x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
        return x_pe, x


class SimilarityLoss(nn.Module):
    def __init__(self, args):
        super(SimilarityLoss, self).__init__()
        self.args = args

    def get_metric(self, metric_type):
        METRICS = {
            'euclidean': lambda gallery, query: torch.sum((query[:, :, None, :, :] - gallery[:, None, :, :, :]) ** 2, (3, 4)),
        }
        return METRICS[metric_type]

    def forward(self, out_f, labels_support, labels_query, att_loss):
        # att_loss_support = att_loss[:self.args.n_way*self.args.n_shot]
        # att_loss_query = att_loss[:self.args.n_way*self.args.n_shot]
        b = labels_query.size()[0]
        out_support = out_f[:b*self.args.n_way*self.args.n_shot, :, :]
        out_query = out_f[b*self.args.n_way*self.args.n_shot:, :, :]

        out_support = out_support.reshape(b, self.args.n_way, self.args.n_shot, self.args.n_way, -1).mean(2)
        out_query = out_query.reshape(b, self.args.n_way*self.args.query, self.args.n_way, -1)

        difference = self.get_metric('euclidean')(out_support, out_query)
        logits = F.log_softmax(-difference, dim=2)
        logits = logits.reshape(b, self.args.query, self.args.n_way, -1)
        labels_query = labels_query.reshape(b, self.args.query, self.args.n_way, -1)
        loss = -logits.gather(3, labels_query).squeeze().view(-1).mean()

        _, y_hat = logits.max(3)
        acc = torch.eq(y_hat, labels_query.squeeze()).float().mean()
        return loss, acc
