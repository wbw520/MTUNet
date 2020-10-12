from model.model_tools import fix_parameter, load_backbone, Trans
import torch.nn as nn
import torch
# from args_setting import *
# from loaders.base_loader import make_loaders
import torch.nn.functional as F
import numpy as np
from model.scouter.scouter_model import SlotModel


def load_slot(args):
    args.use_slot = True
    model = SlotModel(args)
    model_name = f"{args.dataset}_" + 'use_slot_no_fsl_checkpoint.pth'
    checkpoint = torch.load(f"{args.output_dir}/" + model_name, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    init_slot = torch.squeeze(model.state_dict()["slot.initial_slots"], dim=0)
    bone = model.backbone
    conv1x1 = model.conv1x1
    to_k = model.slot.to_k
    gru = model.slot.gru
    fix_parameter(bone, [""], mode="fix")
    fix_parameter(conv1x1, [""], mode="fix")
    fix_parameter(to_k, [""], mode="fix")
    fix_parameter(gru, [""], mode="fix")
    position_emb = model.position_emb
    return init_slot, conv1x1, to_k, gru, position_emb, bone


class FSLPre(nn.Module):
    def __init__(self, args):
        super(FSLPre, self).__init__()
        self.args = args
        self.init_slot, self.conv1x1, self.to_k, self.gru, self.position_emb, self.extractor = load_slot(args)
        self.init_slot = self.init_slot.to(args.device)
        dim = args.hidden_dim
        self.iters = 3
        self.scale = dim ** -0.5

    def get_slot(self, split_index, mode):
        if mode == "train":
            new_slot = []
            selected_slot = []
            for i in range(64):
                if i not in split_index:
                    new_slot.append(torch.unsqueeze(self.init_slot[i], dim=0))
                else:
                    selected_slot.append(torch.unsqueeze(self.init_slot[i], dim=0))
            new_slots = torch.cat(new_slot, dim=0)
            selected_slots = torch.cat(selected_slot, dim=0)
            return [new_slots, selected_slots]
        else:
            new_slots = self.init_slot
            return [new_slots]

    def forward(self, support, split_index, mode):
        x_pe, x = self.feature_deal(support)
        out = self.get_slot(split_index, mode)
        slots = out[0]
        att, pred = self.make_iter(slots, x_pe, x)
        created_slot = self.get_fsl_slot(att, slots)
        x_list = list(x.split(self.args.n_shot, dim=0))
        list_for_x = []
        for i in range(len(x_list)):
            list_for_x.append(torch.unsqueeze(x_list[i], dim=0))
        avg_feature = torch.mean(torch.cat(list_for_x, dim=0), dim=(1, 3), keepdim=False)
        # avg_feature = avg_feature.expand(self.args.n_way, -1)
        if mode != "train":
            return [torch.cat([created_slot, avg_feature], dim=1), x_pe, x]
        else:
            return [torch.cat([created_slot, avg_feature], dim=1), x_pe, x, out[1]]

    def feature_deal(self, x):
        x = self.extractor(x)
        x = self.conv1x1(x)
        pe = self.position_emb(x)
        x_pe = x + pe
        b, n, r, c = x.shape
        x = x.reshape((b, n, -1)).permute((0, 2, 1))
        x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
        return x_pe, x

    def get_fsl_slot(self, att, slots):
        att_list = list(att.split(self.args.n_shot, dim=0))
        selected_slot = []
        for i in range(len(att_list)):
            current_att = att_list[i]
            att_max = torch.argmax(torch.sum(current_att, dim=(0, 2), keepdim=False))
            selected_slot.append(torch.unsqueeze(slots[att_max], dim=0))
        return torch.cat(selected_slot, dim=0)

    def make_iter(self, slots, inputs_pe, inputs_x):
        slots = torch.unsqueeze(slots, dim=0)
        b, n, d = inputs_pe.shape
        slots = slots.expand(b, -1, -1)
        k, v = self.to_k(inputs_pe), inputs_pe
        for _ in range(self.iters):
            slots_prev = slots
            q = slots
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots = torch.div(dots, dots.sum(2).expand_as(dots.permute([2,0,1])).permute([1,2,0])) * dots.sum(2).sum(1).expand_as(dots.permute([1,2,0])).permute([2,0,1])# * 10
            attn = torch.sigmoid(dots)
            updates = torch.einsum('bjd,bij->bid', inputs_x, attn)
            updates = updates / inputs_x.size(2)
            slots, _ = self.gru(
                updates.reshape(1, -1, d),
                slots_prev.reshape(1, -1, d)
            )
            slots = slots.reshape(b, -1, d)
        slot_att = attn
        pred = torch.sum(updates, dim=2, keepdim=False)
        return slot_att, pred


class FSL(FSLPre):
    def __init__(self, args):
        super(FSL, self).__init__(args)
        self.args = args
        # self.af = 0.1
        self.lambda_value = args.lambda_value
        self.mlp = nn.Sequential(
            # nn.ReLU(inplace=True),
            nn.Linear(164, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 64),
        )

    # def get_gausi(self, slots, size):
    #     gau = torch.randn(size, requires_grad=False)
    #     gau = gau.cuda()
    #     slots = slots + (self.af**0.5)*gau
    #     return slots

    def forward(self, slots, supports, query):
        # slots = F.normalize(slots, dim=2)
        # slots = self.get_gausi(slots, slots.size())
        adjusted_slots = self.mlp(slots)
        batch = adjusted_slots.size()[0]
        x, x_pe = self.feature_deal(query)
        if self.training:
            query_num = self.args.n_way_train
        else:
            query_num = self.args.n_way
        x_list = list(x.split(self.args.query*query_num, dim=0))
        x_pe_list = list(x_pe.split(self.args.query*query_num, dim=0))
        att_list_s = []
        output_list_s = []
        att_list_q = []
        output_list_q = []
        for i in range(batch):
            att_q, pre_q = self.make_iter(adjusted_slots[i], x_pe_list[i], x_list[i])
            att_s, pre_s = self.make_iter(adjusted_slots[i], supports[0][i], supports[1][i])
            att_list_s.append(att_s)
            output_list_s.append(pre_s)
            att_list_q.append(att_q)
            output_list_q.append(pre_q)
        return [output_list_s, att_list_s, output_list_q, att_list_q, adjusted_slots]


class FSLLoss(nn.CrossEntropyLoss):
    def __init__(self, args):
        super(FSLLoss, self).__init__()
        self.args = args
        self.power = 2
        self.lamb = float(self.args.lambda_value)

    def one_hot(self, batch_size, label):
        return torch.zeros(batch_size, self.args.n_way).to(self.args.device).scatter_(1, label, 1)

    def forward(self, preds, target):
        output_list_s, att_list_s, output_list_q, att_list_q, adjusted_slots = preds[0]
        slots_true, mode = preds[1], preds[2]
        label_s, label_q = target
        output_s = torch.cat(output_list_s, dim=0)
        output_q = torch.cat(output_list_q, dim=0)
        loss_logits_s = super(FSLLoss, self).forward(output_s, label_s)
        loss_logits_q = super(FSLLoss, self).forward(output_q, label_q)
        # att_s = torch.cat(att_list_s, dim=0)
        # att_q = torch.cat(att_list_q, dim=0)
        # batch_s = att_s.size()[0]
        # batch_q = att_q.size()[0]
        #
        # att_s_l = att_s.clone() * torch.unsqueeze(self.one_hot(batch_s, torch.unsqueeze(label_s, dim=-1)), dim=-1)
        # att_q_l = att_q.clone() * torch.unsqueeze(self.one_hot(batch_q, torch.unsqueeze(label_q, dim=-1)), dim=-1)
        # one_s = torch.ones_like(att_s_l)
        # att_s_l = torch.where(att_s_l > 0.1, one_s, att_s_l)
        # one_q = torch.ones_like(att_q_l)
        # att_q_l = torch.where(att_q_l > 0.1, one_q, att_q_l)
        #
        # loss_ojld_s = torch.mean(F.pairwise_distance(att_s, att_s_l))
        # loss_ojld_q = torch.mean(F.pairwise_distance(att_q, att_q_l))
        slot_loss = 0
        if mode == "train":
            slot_loss = torch.mean(F.pairwise_distance(slots_true, adjusted_slots))
        loss_att_area_s = torch.pow(torch.mean(torch.cat(att_list_s, dim=0)), self.power)
        loss_att_area_q = torch.pow(torch.mean(torch.cat(att_list_q, dim=0)), self.power)

        total_loss = loss_logits_s + loss_logits_q
        # + loss_ojld_s + loss_ojld_q

        return [total_loss, loss_logits_s, loss_logits_q, 0, 0, loss_att_area_s, loss_att_area_q, slot_loss]


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     img = torch.randn(5, 3, 84, 84)
#     img = img.cuda()
#     querys = torch.randn(5*75, 3, 84, 84)
#     qqq = torch.randn(5, 5, 185)
#     split = np.random.permutation(64)[:5]
#     wbw = FSLPre(args)
# #     wbw2 = FSL(args)
#     wbw.cuda()
#     wbw.eval()
#     out = wbw(img, split, "train")
# #     # oo = wbw2(qqq, querys)