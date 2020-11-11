import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import tools.calculate_tool as cal
from torch.autograd import Variable


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def metric_prediction(gallery, query, train_label, metric_type):
    gallery = gallery.view(gallery.shape[0], -1)
    query = query.view(query.shape[0], -1)
    distance = get_metric(metric_type)(gallery, query)
    predict = torch.argmin(distance, dim=1)
    predict = torch.take(train_label, predict)
    return predict


def train_one_epoch(model, data_loader, device, record, epoch, optimizer, criterion):
    model.train()
    L = len(data_loader)
    running_loss = 0.0
    running_corrects_1 = 0.0
    running_corrects_2 = 0.0
    print("start train: " + str(epoch))
    for i, (inputs, target) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device, dtype=torch.float32)
        labels = target.to(device, dtype=torch.int64)

        optimizer.zero_grad()
        logits, feature = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        a = loss.item()
        running_loss += a
        running_corrects_1 += cal.evaluateTop1(logits, labels)
        running_corrects_2 += cal.evaluateTop5(logits, labels)
    record["train"]["loss"].append(round(running_loss/L, 3))
    record["train"]["acc1"].append(round(running_corrects_1/L, 3))
    record["train"]["acc5"].append(round(running_corrects_2/L, 3))


@torch.no_grad()
def evaluate(args, model, data_loader, device, record, epoch):
    model.eval()
    print("start val: " + str(epoch))
    running_corrects_1 = 0.0
    running_acc_95 = []
    L = len(data_loader)
    for i, (inputs, target) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device, dtype=torch.float32)
        labels = target.to(device, dtype=torch.int64)
        logits, feature = model(inputs)

        feature_s = feature[:args.n_way*args.n_shot, :].reshape(args.n_way, args.n_shot, -1).mean(1)
        feature_q = feature[args.n_way*args.n_shot:, :]
        labels_support = Variable(torch.arange(0, args.n_way).long().cuda(), requires_grad=False).reshape(-1)
        labels_query = Variable(torch.arange(0, args.n_way).view(args.n_way, 1).expand(args.n_way, args.query).long().cuda(), requires_grad=False).reshape(-1)
        prediction = metric_prediction(feature_s, feature_q, labels_support, 'euclidean')
        acc = (prediction == labels_query).float().mean()
        running_corrects_1 += acc.item()
        running_acc_95.append(round(acc.item(), 4))
    record["val"]["acc1"].append(round(running_corrects_1/L, 3))
    record["val"]["accm"].append(round(cal.compute_confidence_interval(running_acc_95)[0], 3))
    record["val"]["accpm"].append(round(cal.compute_confidence_interval(running_acc_95)[1], 3))
