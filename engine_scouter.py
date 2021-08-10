import torch
import tools.calculate_tool as cal
from tqdm.auto import tqdm


def train_one_epoch(args, model, data_loader, device, record, epoch, optimizer):
    calculation(args, model, "train", data_loader, device, record, epoch, optimizer)



@torch.no_grad()
def evaluate(args, model, data_loader, device, record, epoch):
    calculation(args, model, "val", data_loader, device, record, epoch)


def calculation(args, model, mode, data_loader, device, record, epoch, optimizer=None):
    if mode == "train":
        model.train()
    else:
        model.eval()
    L = len(data_loader)
    running_loss = 0.0
    running_corrects = 0.0
    running_att_loss = 0.0
    running_log_loss = 0.0
    print("start " + mode + " :" + str(epoch))
    if optimizer is not None:
        print("current learning rate: ", optimizer.param_groups[0]["lr"])
    for i, (inputs, target) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device, dtype=torch.float32)
        labels = target.to(device, dtype=torch.int64)

        if mode == "train":
            optimizer.zero_grad()
        logits, loss_list = model(inputs, labels)
        loss = loss_list[0]
        if mode == "train":
            loss.backward()
            optimizer.step()

        a = loss.item()
        running_loss += a
        if len(loss_list) > 2:  # For slot training only
            running_att_loss += loss_list[2].item()
            running_log_loss += loss_list[1].item()
        running_corrects += cal.evaluateTop1(logits, labels)
    epoch_loss = round(running_loss/L, 3)
    epoch_loss_log = round(running_log_loss/L, 3)
    epoch_loss_att = round(running_att_loss/L, 3)
    epoch_acc = round(running_corrects/L, 3)
    record[mode]["loss"].append(epoch_loss)
    record[mode]["acc"].append(epoch_acc)
    record[mode]["log_loss"].append(epoch_loss_log)
    record[mode]["att_loss"].append(epoch_loss_att)