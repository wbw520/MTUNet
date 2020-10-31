import torch
import torch.nn as nn
import torch.nn.functional as F
from model.scouter.scouter_attention import ScouterAttention
from model.scouter.position_encode import build_position_encoding
from model.model_tools import fix_parameter, load_backbone


class SlotModel(nn.Module):
    def __init__(self, args):
        super(SlotModel, self).__init__()
        self.use_slot = args.use_slot
        self.backbone = load_backbone(args)
        if self.use_slot:
            self.channel = args.channel
            self.slots_per_class = args.slots_per_class
            self.conv1x1 = nn.Conv2d(self.channel, args.hidden_dim, kernel_size=(1, 1), stride=(1, 1))
            if args.fix_parameter:
                fix_parameter(self.backbone, [""], mode="fix")
                # fix_parameter(self.backbone, ["layer4", "layer3"], mode="open")
            self.slot = ScouterAttention(args, self.slots_per_class, args.hidden_dim, vis=args.vis,
                    vis_id=args.vis_id, loss_status=args.loss_status, power=args.power, to_k_layer=args.to_k_layer)
            self.position_emb = build_position_encoding('sine', hidden_dim=args.hidden_dim)
            self.lambda_value = float(args.lambda_value)

    def forward(self, x, target=None):
        x = self.backbone(x)
        if self.use_slot:
            x = self.conv1x1(x)
            x = torch.relu(x)
            pe = self.position_emb(x)
            x_pe = x + pe

            b, n, r, c = x.shape
            x = x.reshape((b, n, -1)).permute((0, 2, 1))
            x_pe = x_pe.reshape((b, n, -1)).permute((0, 2, 1))
            x, attn_loss = self.slot(x_pe, x)
        output = F.log_softmax(x, dim=1)

        if target is not None:
            if self.use_slot:
                loss = F.nll_loss(output, target) + self.lambda_value * attn_loss
                return [output, [loss, F.nll_loss(output, target), attn_loss]]
            else:
                loss = F.nll_loss(output, target)
                return [output, [loss]]

        return output