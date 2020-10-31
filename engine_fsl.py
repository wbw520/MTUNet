import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import tools.calculate_tool as cal


def train_one_epoch(model, data_loader, device, record, epoch, optimizer, criterion):
    model.train()
    criterion.train()
    L = len(data_loader)
    running_loss = 0.0
    running_att_loss = 0.0
    running_acc_95 = []
    print("start train: " + str(epoch))
    print(optimizer.param_groups[0]["lr"])
    for i, (inputs, target) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device, dtype=torch.float32)
        optimizer.zero_grad()
        out, att_loss = model(inputs)
        loss, acc, logits = criterion(out, att_loss)
        loss.backward()
        optimizer.step()

        a = loss.item()
        running_loss += a
        running_att_loss += att_loss.item()
        running_acc_95.append(round(acc.item(), 4))

    record["train"]["loss"].append(round(running_loss/L, 3))
    record["train"]["att_loss"].append(round(running_att_loss/L, 6))
    record["train"]["accm"].append(round(cal.compute_confidence_interval(running_acc_95)[0], 4))
    record["train"]["accpm"].append(round(cal.compute_confidence_interval(running_acc_95)[1], 4))


@torch.no_grad()
def evaluate(model, data_loader, device, record, epoch, criterion):
    model.eval()
    criterion.eval()
    print("start val: " + str(epoch))
    running_loss = 0.0
    running_att_loss = 0.0
    running_acc_95 = []
    L = len(data_loader)
    for i, (inputs, target) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device, dtype=torch.float32)

        out, att_loss = model(inputs)
        loss, acc, logits = criterion(out, att_loss)
        a = loss.item()
        running_loss += a
        running_att_loss += att_loss.item()
        running_acc_95.append(round(acc.item(), 4))

    record["val"]["loss"].append(round(running_loss/L, 3))
    record["val"]["att_loss"].append(round(running_att_loss/L, 6))
    record["val"]["accm"].append(round(cal.compute_confidence_interval(running_acc_95)[0], 4))
    record["val"]["accpm"].append(round(cal.compute_confidence_interval(running_acc_95)[1], 4))
