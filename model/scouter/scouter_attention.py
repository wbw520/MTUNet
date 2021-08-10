from torch import nn
import torch
from PIL import Image
import numpy as np


class ScouterAttention(nn.Module):
    def __init__(self, args, slots_per_class, dim, iters=3, eps=1e-8, vis=False, vis_id=0, loss_status=1, power=1, to_k_layer=1):
        super().__init__()
        self.args = args
        self.slots_per_class = slots_per_class
        self.num_slots = self.args.num_slot * slots_per_class
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.loss_status = loss_status

        slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, dim)))
        mu = slots_mu.expand(1, self.num_slots, -1)
        sigma = slots_sigma.expand(1, self.num_slots, -1)
        self.initial_slots = nn.Parameter(torch.normal(mu, sigma))

        to_k_layer_list = [nn.Linear(dim, dim)]
        for to_k_layer_id in range(1, to_k_layer):
            to_k_layer_list.append(nn.ReLU(inplace=True))
            to_k_layer_list.append(nn.Linear(dim, dim))

        self.to_k = nn.Sequential(
            *to_k_layer_list
        )
        self.gru = nn.GRU(dim, dim)
        self.vis = vis
        self.vis_id = vis_id
        self.power = power

    def forward(self, inputs, inputs_x):
        b, n, d = inputs.shape
        slots = self.initial_slots.expand(b, -1, -1)
        k, v = self.to_k(inputs), inputs_x
        for _ in range(self.iters):
            slots_prev = slots
            q = slots

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            dots = torch.div(dots, dots.sum(2).expand_as(dots.permute([2,0,1])).permute([1,2,0])) * dots.sum(2).sum(1).expand_as(dots.permute([1,2,0])).permute([2,0,1])# * 10
            if self.args.slot_base_train:
                attn = torch.sigmoid(dots)
            elif self.args.double:
                attn1 = dots.softmax(dim=1)
                attn2 = dots.softmax(dim=1)
                attn = attn1*attn2
            else:
                attn1 = dots.softmax(dim=1)
                attn2 = dots.sigmoid()
                attn = attn1*attn2
            updates = torch.einsum('bjd,bij->bid', v, attn)
            updates = updates / v.size(2)
            slots, _ = self.gru(
                updates.reshape(1, -1, d),
                slots_prev.reshape(1, -1, d)
            )
            slots = slots.reshape(b, -1, d)

            if self.vis:
                slots_vis_raw = attn.clone()

        if self.vis:
            b = slots_vis_raw.size()[0]
            for i in range(b):
                slots_vis = slots_vis_raw[i]
                slots_vis = ((slots_vis - slots_vis.min()) / (slots_vis.max()-slots_vis.min()) * 255.).reshape(slots_vis.shape[:1]+(int(slots_vis.size(1)**0.5), int(slots_vis.size(1)**0.5)))
                slots_vis = (slots_vis.cpu().detach().numpy()).astype(np.uint8)
                for id, image in enumerate(slots_vis):
                    image = Image.fromarray(image, mode='L').resize((self.args.img_size, self.args.img_size), resample=Image.BILINEAR)
                    image.save(f'vis/att/{i}_slot_{id:d}.png')

        attn_relu = torch.relu(attn)
        slot_loss = torch.mean(attn_relu, (0, 1, 2))  # * self.slots_per_class

        if self.args.fsl:
            return updates, torch.pow(slot_loss, self.power), attn
        else:
            return self.loss_status*torch.sum(updates, dim=2, keepdim=False), torch.pow(slot_loss, self.power)


