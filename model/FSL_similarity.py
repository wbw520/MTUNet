import torch
import model.extractor as base_bone
from model.model_tools import Identical, fix_parameter
from model.scouter.scouter_attention import ScouterAttention
from model.scouter.position_encode import build_position_encoding
import torch.nn as nn
import torch.nn.functional as F


def load_base(args):
    bone = base_bone.__dict__[args.base_model](num_classes=args.num_classes, drop_dim=False, extract=False)
    # checkpoint = torch.load(f"{args.output_dir}/" + "simple_few_checkpoint.pth", map_location=args.device)
    # bone.load_state_dict(checkpoint["model"])
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
        self.slot = ScouterAttention(args, args.n_way, self.slots_per_class, args.hidden_dim, vis=args.vis,
                                     vis_id=args.vis_id, loss_status=args.loss_status, power=args.power, to_k_layer=args.to_k_layer)
        fix_parameter(self.conv1x1, [""], mode="fix")
        fix_parameter(self.slot, [""], mode="fix")
        self.position_emb = build_position_encoding('sine', hidden_dim=args.hidden_dim)
        self.lambda_value = float(args.lambda_value)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                                        nn.LayerNorm(1024),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048, 2048),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Linear(2048, 1),
                                        nn.Sigmoid(),
        )
        # self.att_query = nn.Sequential(nn.Linear(128, 128),
        #                                nn.ReLU(),
        #                                nn.Linear(128, 128),
        #                                nn.ReLU(),
        #                                nn.Linear(128, 1))

    def forward(self, x):
        b = self.args.batch_size
        x_pe, x, x_raw = self.feature_deal(x)
        x, attn_loss, attn = self.slot(x_pe, x)
        # out_support = x[:b*self.args.n_way*self.args.n_shot, :, :]
        # out_query = x[b*self.args.n_way*self.args.n_shot:, :, :]
        attn_support = attn[:b*self.args.n_way*self.args.n_shot, :, :]
        attn_query = attn[b*self.args.n_way*self.args.n_shot:, :, :]
        x_raw_support = x_raw[:b*self.args.n_way*self.args.n_shot]
        x_raw_query = x_raw[b*self.args.n_way*self.args.n_shot:]
        size = x_raw_support.size()[-1]
        dim = x_raw_support.size()[1]

        attn_support = attn_support.mean(1, keepdim=True).reshape(b, self.args.n_way, 1, size, size)
        attn_query = attn_query.mean(1, keepdim=True).reshape(b, self.args.n_way*self.args.query, 1, size, size)
        weighted_support = torch.mean(attn_support*(x_raw_support.reshape(b, self.args.n_way, dim, size, size)), dim=(3, 4))
        weighted_query = torch.mean(attn_query*(x_raw_query.reshape(b, self.args.n_way*self.args.query, dim, size, size)), dim=(3, 4))

        input_fc = torch.cat(
            [weighted_support.unsqueeze(1).expand(-1, self.args.n_way*self.args.query, -1, -1),
             weighted_query.unsqueeze(2).expand(-1, -1, self.args.n_way, -1)],
            dim=-1
        )

        out_fc = self.classifier(input_fc).squeeze(-1)
        return out_fc, attn_loss
        
    def feature_deal(self, x):
        x_raw = self.backbone(x)
        x = self.conv1x1(x_raw)
        pe = self.position_emb(x)
        x_pe = x + pe
        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0, 2, 1))
        x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
        return x_pe, x, x_raw


class SimilarityLoss(nn.Module):
    def __init__(self, args):
        super(SimilarityLoss, self).__init__()
        self.args = args
        self.BCEloss = nn.BCELoss()

    def get_slots(self, input, max):
        return torch.gather(input, 3, max)

    def forward(self, out_fc, labels_support, labels_query, att_loss, mode):
        labels_query_onehot = torch.zeros(labels_query.size()+(5,), dtype=labels_query.dtype).to(labels_query.device)
        labels_query_onehot.scatter_(-1, labels_query.unsqueeze(-1), 1)
        BCELoss = self.BCEloss(out_fc, labels_query_onehot.float())
        loss = BCELoss + float(self.args.lambda_value) * att_loss
        logits = F.log_softmax(out_fc, dim=-1)
        _, y_hat = logits.max(2)
        acc = torch.eq(y_hat, labels_query).float().mean()
        return loss, acc