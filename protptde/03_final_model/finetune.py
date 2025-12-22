import json
import torch
import numpy as np
import pandas as pd
from model import BatchData, to_gpu, spearman_corr
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def stratified_sampling_for_mutation_data(mut_info_list):
    positions = set()
    for multiple_mut_info in mut_info_list:
        for single_mut_info in multiple_mut_info.split(","):
            positions.add(int(single_mut_info[1:-1]))
    sorted_mut_positions = sorted(positions)
    index_map = {mut_pos: index for index, mut_pos in enumerate(sorted_mut_positions)}
    vectors_dict = {}
    for multiple_mut_info in mut_info_list:
        vec = [0] * len(index_map)
        for single_mut_info in multiple_mut_info.split(","):
            vec[index_map[int(single_mut_info[1:-1])]] = 1
        vectors_dict[multiple_mut_info] = vec
    return sorted_mut_positions, index_map, vectors_dict


basic_data_name = config["basic_data_name"]
best_hyperparameters = config["best_hyperparameters"]
finetune_parameter = config["final_model"]["finetune_parameter"]

selected_models = best_hyperparameters["selected_models"]
num_layer = best_hyperparameters["num_layer"]
random_seed = best_hyperparameters["random_seed"]

max_lr = finetune_parameter["finetune_max_lr"]
device = finetune_parameter["device"]
min_lr = finetune_parameter["min_lr"]
initial_lr = finetune_parameter["initial_lr"]
total_epochs = finetune_parameter["total_epochs"]
warmup_epochs = int(finetune_parameter["warmup_epochs_ratio"] * total_epochs)
batch_size = finetune_parameter["batch_size"]
test_size = finetune_parameter["test_size"]

all_csv = pd.read_csv(f"../01_data_processing/{basic_data_name}.csv", index_col=0)
mut_info_list = all_csv.index.tolist()
_, _, vectors = stratified_sampling_for_mutation_data(mut_info_list)

y_mut_pos = np.array([vectors[multiple_mut_info] for multiple_mut_info in mut_info_list])
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
for train_index, test_index in msss.split(y_mut_pos, y_mut_pos):
    train_csv = all_csv.iloc[train_index].copy()
    test_csv = all_csv.iloc[test_index].copy()

train_dataset = BatchData(train_csv, selected_models)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
test_dataset = BatchData(test_csv, selected_models)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

model = torch.load(f"results/{random_seed}/train_best.pt", map_location=lambda storage, loc: storage.cuda(device), weights_only=False)

for name, param in model.named_parameters():
    if "finetune_coef" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=initial_lr)
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs) * (max_lr / initial_lr))
plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, min_lr=min_lr)

best_loss = float("inf")
loss = pd.DataFrame()

for epoch in range(total_epochs):

    model.train()
    epoch_loss = 0
    for wt_data, mut_data, label in train_loader:
        wt_data, mut_data, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(label, device)
        optimizer.zero_grad()
        pred = model(wt_data, mut_data)
        train_loss = torch.nn.functional.mse_loss(pred, label)
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
        optimizer.step()
        epoch_loss += train_loss.item()
    train_loss = epoch_loss / len(train_loader)

    preds = []
    trues = []
    with torch.no_grad():
        for wt_data, mut_data, label in test_loader:
            wt_data, mut_data, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(label, device)
            pred = model(wt_data, mut_data)
            preds += pred.detach().cpu().tolist()
            trues += label.detach().cpu().tolist()
    test_corr = spearman_corr(torch.tensor(preds), torch.tensor(trues)).item()
    test_loss = torch.nn.functional.mse_loss(torch.tensor(preds), torch.tensor(trues)).item()

    loss.loc[f"{epoch}", "train_loss"] = train_loss
    loss.loc[f"{epoch}", "test_corr"] = test_corr
    loss.loc[f"{epoch}", "test_loss"] = test_loss
    loss.loc[f"{epoch}", "learning_rate"] = optimizer.param_groups[0]["lr"]
    loss.to_csv(f"results/{random_seed}/finetune_loss.csv")
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, test_corr={test_corr:.4f}, test_loss={test_loss:.4f}, learning_rate={optimizer.param_groups[0]['lr']:.6f}")

    if epoch < warmup_epochs:
        warmup_scheduler.step()
    else:
        plateau_scheduler.step(test_loss)

    if test_loss < best_loss:
        best_loss = test_loss
        torch.save(model.state_dict(), "finetune_best.pth")
    elif optimizer.param_groups[0]["lr"] <= min_lr and epoch > warmup_epochs:
        print(f"Stopping at epoch {epoch} due to no improvement in test loss.")
        break
