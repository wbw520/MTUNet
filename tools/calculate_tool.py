import torch
import numpy as np
import scipy.stats
import scipy as sp


def evaluateTop1(logits, labels):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        # print("eval_Pred", pred)
        # print("eval_Lbale", labels)
        return torch.eq(pred, labels).sum().float().item()/labels.size(0)


def evaluateTop5(logits, labels):
    with torch.no_grad():
        maxk = max((1, 5))
        labels_resize = labels.view(-1, 1)
        _, pred = logits.topk(maxk, 1, True, True)
        return torch.eq(pred, labels_resize).sum().float().item()/labels.size(0)


def compute_confidence_interval(data):
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


class MetricLog():
    def __init__(self, args):
        self.args = args
        self.record = {"train": {"loss": [], "acc": [], "log_loss": [], "att_loss": []},
                       "val": {"loss": [], "acc": [], "log_loss": [], "att_loss": []}}

    def print_metric(self):
        print("train loss:", self.record["train"]["loss"])
        print("val loss:", self.record["val"]["loss"])
        print("train acc:", self.record["train"]["acc"])
        print("val acc:", self.record["val"]["acc"])
        print("train CE loss", self.record["train"]["log_loss"])
        print("val CE loss", self.record["val"]["log_loss"])
        print("train attention loss", self.record["train"]["att_loss"])
        print("val attention loss", self.record["val"]["att_loss"])
        return self.record


class MetricLogSimilar():
    def __init__(self, args):
        self.args = args
        self.record = {"train": {"loss": [], "accm": [], "accpm": [], "att_loss": []},
                       "val": {"loss": [], "accm": [], "accpm": [], "att_loss": []}}

    def print_metric(self):
        print("train loss:", self.record["train"]["loss"])
        print("train att loss", self.record["train"]["att_loss"])
        print("train accm:", self.record["train"]["accm"])
        print("train accpm", self.record["train"]["accpm"])
        print("val loss:", self.record["val"]["loss"])
        print("val att loss", self.record["val"]["att_loss"])
        print("val accm", self.record["val"]["accm"])
        print("val accpm", self.record["val"]["accpm"])


class MetricLogFew():
    def __init__(self, args):
        self.args = args
        self.record = {"train": {"loss": [], "acc1": [], "acc5": []},
                       "val": {"acc1": [], "accm": [], "accpm": []}}

    def print_metric(self):
        print("train loss:", self.record["train"]["loss"])
        print("train acc1:", self.record["train"]["acc1"])
        print("train acc5", self.record["train"]["acc5"])
        print("val acc1", self.record["val"]["acc1"])
        print("val accm", self.record["val"]["accm"])
        print("val accpm", self.record["val"]["accpm"])