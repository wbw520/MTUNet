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
            'euclidean': lambda gallery, query: torch.sum((query[:, :, None, :] - gallery[:, None, :, :]) ** 2, 3),
        }
        return METRICS[metric_type]
    def get_selected(self, data, index):
        record_list = []
        for i in range(len(index)):
            record_list.append(data[i, :, index[i], :])
        return torch.cat(record_list, dim=0)
    def cover_select(self, input):
        index_list = []
        b = input.size()[0]
        cls = input.size()[1]
        for i in range(b):
            temp_index = []
            sprted_slot, indices_slot = torch.sort(input[i], descending=True)
            for j in range(cls):
                ss = 0
                while not (indices_slot[j][ss] not in temp_index):
                    ss += 1
                temp_index.append(indices_slot[j][ss])
                index_list.append(indices_slot[j][ss].unsqueeze(0))
        return torch.cat(index_list, dim=0)
    def get_22(self, input, b, max):
        input = input.reshape(b, self.args.n_way*self.args.query, self.args.num_slot, -1)
        imgs = input.size()[1]
        record_list = []
        for j in range(b):
            record_temp = []
            for s in range(imgs):
                for i in range(len(max[j])):
                    record_temp.append(input[j, s, max[j][i], :].unsqueeze(0))
                new_temp = torch.cat(record_temp, dim=0)
                index_max = torch.argmax(torch.mean(new_temp, dim=1))
                record_list.append(new_temp[index_max, :].unsqueeze(0))
        return torch.cat(record_list, dim=0)
    def forward(self, out_f, labels_support, labels_query, att_loss, mode):
        # att_loss_support = att_loss[:self.args.n_way*self.args.n_shot]
        # att_loss_query = att_loss[:self.args.n_way*self.args.n_shot]
        b = labels_query.size()[0]
        out_support = out_f[:b*self.args.n_way*self.args.n_shot, :, :]
        out_query = out_f[b*self.args.n_way*self.args.n_shot:, :, :]
        out_support = out_support.reshape(b*self.args.n_way, self.args.n_shot, self.args.num_slot, -1).mean(1)
        out_support = torch.unsqueeze(out_support, dim=1)
        # max_s = torch.argmax(out_support.mean(3), dim=2).reshape(-1)
        max_s = self.cover_select(out_support.mean(3).reshape(b, self.args.n_way, self.args.num_slot))
        out_query = out_query.reshape(b*self.args.n_way, self.args.query, self.args.num_slot, -1)
        out_support = self.get_selected(out_support, max_s.reshape(-1)).reshape(b, self.args.n_way, -1)
        if mode == "train":
            temp = out_query.clone()
            out_query = self.get_selected(out_query, max_s.reshape(-1)).reshape(b, self.args.n_way*self.args.query, -1)
            out_query_t = self.get_22(temp, b, max_s.reshape(b, -1)).reshape(b, self.args.n_way*self.args.query, -1)
        else:
            out_query = self.get_22(out_query, b, max_s.reshape(b, -1)).reshape(b, self.args.n_way*self.args.query, -1)
        difference = self.get_metric('euclidean')(out_support, out_query)
        logits = F.log_softmax(-difference, dim=2)
        logits = logits.reshape(b, self.args.query, self.args.n_way, -1)
        labels_query = labels_query.reshape(b, self.args.query, self.args.n_way, -1)
        loss = -logits.gather(3, labels_query).squeeze().view(-1).mean() + float(self.args.lambda_value) * att_loss
        _, y_hat = logits.max(3)
        acc = torch.eq(y_hat, labels_query.squeeze()).float().mean()
        if mode == "train":
            difference_t = self.get_metric('euclidean')(out_support, out_query_t)
            logits_t = F.log_softmax(-difference_t, dim=2)
            logits_t = logits_t.reshape(b, self.args.query, self.args.n_way, -1)
            loss_t = -logits_t.gather(3, labels_query).squeeze().view(-1).mean()
            _, y_hat_t = logits_t.max(3)
            acc_t = torch.eq(y_hat_t, labels_query.squeeze()).float().mean()
        if mode == "train":
            return [loss, loss_t], [acc, acc_t]
        else:
            return loss, acc