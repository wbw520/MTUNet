import torch
import tools.calculate_tool as cal
from tqdm.auto import tqdm


def train_one_epoch(args, model, data_loader, device, record, epoch, optimizer, criterien=None):
    if args.fsl:
        fsl_cal(args, model, "train", data_loader, device, record, epoch, optimizer, criterien=criterien)
    else:
        calculation(args, model, "train", data_loader, device, record, epoch, optimizer, criterien=criterien)


@torch.no_grad()
def evaluate(args, model, data_loader, device, record, epoch, criterien=None):
    if args.fsl:
        fsl_cal(args, model, "val", data_loader, device, record, epoch, criterien=criterien)
    else:
        calculation(args, model, "val", data_loader, device, record, epoch, criterien=criterien)


def calculation(args, model, mode, data_loader, device, record, epoch, optimizer=None, criterien=None):
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
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)

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


def fsl_cal(args, model_list, mode, data_loader, device, record, epoch, optimizer=None, criterien=None):
    model_creation = model_list[0]
    model_mlp = model_list[1]
    if mode == "train":
        model_mlp.train()
    else:
        model_mlp.eval()
    running_loss_total = 0.0
    running_acc_95 = []
    running_log_loss_s = 0.0
    running_log_loss_q = 0.0
    running_ojld_loss_s = 0.0
    running_ojld_loss_q = 0.0
    running_att_loss_s = 0.0
    running_att_loss_q = 0.0
    running_slot_loss = 0.0
    L = len(data_loader)
    print("start " + mode + " :" + str(epoch))
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs_query = torch.cat(list(sample_batch["query"]["image"].to(device, dtype=torch.float32)), dim=0)
        labels_query_ = list(sample_batch["query"]["label"].to(device, dtype=torch.int64))
        labels_query = torch.cat(labels_query_, dim=0)
        inputs_support = sample_batch["support"]["image"].to(device, dtype=torch.float32)
        labels_support = torch.cat(list(sample_batch["support"]["label"].to(device, dtype=torch.int64)), dim=0)
        selected_cls = sample_batch["selected_cls"]
        batch = selected_cls.size()[0]
        if mode == "train":
            optimizer.zero_grad()
        selected_slots_list = []
        created_slot_lists = []
        support_x_list = []
        support_x_pe_list = []
        for i in range(batch):
            out = model_creation(inputs_support[i], selected_cls[i], mode)
            new_slot, s_x, s_x_pe = out[0], out[1], out[2]
            created_slot_lists.append(torch.unsqueeze(new_slot, dim=0))
            support_x_list.append(s_x)
            support_x_pe_list.append(s_x_pe)
            if mode == "train":
                selected_slots_list.append(out[3])
        if mode == "train":
            selected_slots = torch.cat(selected_slots_list, dim=0)
        else:
            selected_slots = []
        created_slot = torch.cat(created_slot_lists, dim=0)
        out_results = model_mlp(created_slot, [support_x_pe_list, support_x_list], inputs_query)
        query_results = out_results[2]

        loss_list = criterien([out_results, selected_slots, mode], [labels_support, labels_query])
        loss_total = loss_list[0]
        if mode == "train":
            loss_total.backward()
            optimizer.step()

        for j in range(batch):
            acc = cal.evaluateTop1(query_results[j], labels_query_[j])
            running_acc_95.append(round(acc, 4))

        running_loss_total += loss_total.item()
        running_log_loss_s += loss_list[1].item()
        running_log_loss_q += loss_list[2].item()
        # running_ojld_loss_s += loss_list[3].item()
        # running_ojld_loss_q += loss_list[4].item()
        running_att_loss_s += loss_list[5].item()
        running_att_loss_q += loss_list[6].item()
        if mode == "train":
            running_slot_loss += loss_list[7].item()

    record[mode]["loss_total"].append(round(running_loss_total/L, 3))
    record[mode]["acc_95"].append(round(cal.confidenceinterval(running_acc_95)[0], 3))
    record[mode]["log_loss_s"].append(round(running_log_loss_s/L, 3))
    record[mode]["log_loss_q"].append(round(running_log_loss_q/L, 3))
    # record[mode]["ojld_loss_s"].append(round(running_ojld_loss_s/L, 3))
    # record[mode]["ojld_loss_q"].append(round(running_ojld_loss_q/L, 3))
    record[mode]["att_loss_s"].append(round(running_att_loss_s/L, 3))
    record[mode]["att_loss_q"].append(round(running_att_loss_q/L, 3))
    if mode == "train":
        record[mode]["slot_loss"].append(round(running_slot_loss/L, 3))