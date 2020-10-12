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


def confidenceinterval(array):
    n = len(array)
    confidence = 0.95
    a = 1.0*np.array(array)
    m = np.mean(a)
    fc = scipy.stats.sem(a)
    h = fc * sp.stats.t._ppf((1+confidence)/2., n-1)/((n-1)**0.5)
    return [m, h]


class MetricLog():
    def __init__(self, args):
        self.args = args
        if args.fsl:
            self.record = {"train": {"loss_total": [], "acc_95": [], "log_loss_s": [], "log_loss_q": [],
                                     "ojld_loss_s": [], "ojld_loss_q": [], "att_loss_s": [], "att_loss_q": [], "slot_loss": []},
                           "val": {"loss_total": [], "acc_95": [], "log_loss_s": [], "log_loss_q": [],
                                   "ojld_loss_s": [], "ojld_loss_q": [], "att_loss_s": [], "att_loss_q": []}}
        else:
            self.record = {"train": {"loss": [], "acc": [], "log_loss": [], "att_loss": []},
                       "val": {"loss": [], "acc": [], "log_loss": [], "att_loss": []}}

    def print_metric(self):
        if self.args.fsl :
            print("train total loss: ", self.record["train"]["loss_total"])
            print("train acc 95: ", self.record["train"]["acc_95"])
            print("train log loss support: ", self.record["train"]["log_loss_s"])
            print("train log loss query: ", self.record["train"]["log_loss_q"])
            print("train ojld loss support: ", self.record["train"]["ojld_loss_s"])
            print("train ojld loss query: ", self.record["train"]["ojld_loss_q"])
            print("train att loss support", self.record["train"]["att_loss_s"])
            print("train att loss query", self.record["train"]["att_loss_q"])
            print("train slot loss", self.record["train"]["slot_loss"])
            print("val total loss: ", self.record["val"]["loss_total"])
            print("val acc 95: ", self.record["val"]["acc_95"])
            print("val log loss support: ", self.record["val"]["log_loss_s"])
            print("val log loss query: ", self.record["val"]["log_loss_q"])
            print("val ojld loss support: ", self.record["val"]["ojld_loss_s"])
            print("val ojld loss query: ", self.record["val"]["ojld_loss_q"])
            print("val att loss support", self.record["val"]["att_loss_s"])
            print("val att loss query", self.record["val"]["att_loss_q"])

        else:
            print("train loss:", self.record["train"]["loss"])
            print("val loss:", self.record["val"]["loss"])
            print("train acc:", self.record["train"]["acc"])
            print("val acc:", self.record["val"]["acc"])
            print("train CE loss", self.record["train"]["log_loss"])
            print("val CE loss", self.record["val"]["log_loss"])
            print("train attention loss", self.record["train"]["att_loss"])
            print("val attention loss", self.record["val"]["att_loss"])
