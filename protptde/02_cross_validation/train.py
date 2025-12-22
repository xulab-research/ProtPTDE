import os
import uuid
import json
import click
import torch
import itertools
import numpy as np
import pandas as pd
from model import BatchData, ModelUnion, to_gpu, spearman_loss, spearman_corr
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit, MultilabelStratifiedKFold


with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def format_float_no_sci_no_trailzero(x):
    if isinstance(x, (float, np.floating, int, np.integer)):
        s = f"{float(x):.12f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
        return s
    return str(x)


def save_csv_no_sci_append(path, new_df, append, dedup_cols=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if append and os.path.exists(path):
        old_df = pd.read_csv(path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        if dedup_cols:
            merged = merged.drop_duplicates(subset=dedup_cols, keep="first")
        merged.to_csv(path, index=False)
    else:
        new_df.to_csv(path, index=False)


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


def objective(trial, random_seed):
    basic_data_name = config["basic_data_name"]
    model_number = config["cross_validation"]["model_number"]
    training_parameter = config["cross_validation"]["training_parameter"]
    hyperparameter_search = config["cross_validation"]["hyperparameter_search"]

    all_models = sorted(list(config["all_model"].keys()))
    model_combinations = [",".join(combo) for combo in itertools.combinations(all_models, model_number)]

    selected_models = trial.suggest_categorical("model_combination", model_combinations).split(",")
    num_layer = trial.suggest_int("num_layer", hyperparameter_search["num_layer"]["min"], hyperparameter_search["num_layer"]["max"])
    max_lr = trial.suggest_categorical("max_lr", hyperparameter_search["max_lr"]["choices"])

    device = training_parameter["device"]
    min_lr = training_parameter["min_lr"]
    initial_lr = training_parameter["initial_lr"]
    total_epochs = training_parameter["total_epochs"]
    warmup_epochs = int(training_parameter["warmup_epochs_ratio"] * total_epochs)
    batch_size = training_parameter["batch_size"]
    test_size = training_parameter["test_size"]
    k_fold = training_parameter["k_fold"]
    cv_shuffle = training_parameter["shuffle"]

    all_csv = pd.read_csv(f"../01_data_processing/{basic_data_name}.csv", index_col=0)
    mut_info_list = all_csv.index.tolist()
    _, _, vectors = stratified_sampling_for_mutation_data(mut_info_list)
    y_mut_pos = np.array([vectors[multiple_mut_info] for multiple_mut_info in mut_info_list])

    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    for train_validation_index, test_index in msss.split(y_mut_pos, y_mut_pos):
        train_validation_csv = all_csv.iloc[train_validation_index].copy()
        test_csv = all_csv.iloc[test_index].copy()

    models_name = ",".join(selected_models)
    file = f"results/{models_name}/num_layer_{num_layer}_max_lr_{max_lr}_random_seed_{random_seed}"
    os.makedirs(file, exist_ok=True)
    k_fold_test_corr = []

    mskf = MultilabelStratifiedKFold(n_splits=k_fold, shuffle=cv_shuffle, random_state=random_seed)
    for k_fold_index, (train_index, validation_index) in enumerate(mskf.split(y_mut_pos[train_validation_index], y_mut_pos[train_validation_index])):
        train_csv = train_validation_csv.iloc[train_index].copy()
        validation_csv = train_validation_csv.iloc[validation_index].copy()
        train_dataset = BatchData(train_csv, selected_models)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        validation_dataset = BatchData(validation_csv, selected_models)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        test_dataset = BatchData(test_csv, selected_models)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

        model = ModelUnion(num_layer, selected_models).to(device)
        for name, param in model.named_parameters():
            if "finetune_coef" in name:
                param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=initial_lr)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min(1.0, epoch / warmup_epochs) * (max_lr / initial_lr))
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, min_lr=min_lr)

        best_loss = float("inf")
        best_corr = float("-inf")
        loss = pd.DataFrame()

        for epoch in range(total_epochs):
            model.train()
            epoch_loss = 0
            for wt_data, mut_data, label in train_loader:
                wt_data, mut_data, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(label, device)
                optimizer.zero_grad()
                pred = model(wt_data, mut_data)
                train_loss = spearman_loss(pred.unsqueeze(0), label.unsqueeze(0), 1e-2, "kl")
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), norm_type=2, max_norm=10, error_if_nonfinite=True)
                optimizer.step()
                epoch_loss += train_loss.item()
            train_loss = epoch_loss / len(train_loader)

            model.eval()
            preds = []
            trues = []
            with torch.no_grad():
                for wt_data, mut_data, label in validation_loader:
                    wt_data, mut_data, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(label, device)
                    pred = model(wt_data, mut_data)
                    preds += pred.detach().cpu().tolist()
                    trues += label.detach().cpu().tolist()
            validation_loss = -spearman_corr(torch.tensor(preds), torch.tensor(trues)).item()

            preds = []
            trues = []
            with torch.no_grad():
                for wt_data, mut_data, label in test_loader:
                    wt_data, mut_data, label = to_gpu(wt_data, device), to_gpu(mut_data, device), to_gpu(label, device)
                    pred = model(wt_data, mut_data)
                    preds += pred.detach().cpu().tolist()
                    trues += label.detach().cpu().tolist()
            test_corr = spearman_corr(torch.tensor(preds), torch.tensor(trues)).item()

            loss.loc[f"{epoch}", "train_loss"] = train_loss
            loss.loc[f"{epoch}", "validation_loss"] = validation_loss
            loss.loc[f"{epoch}", "test_corr"] = test_corr
            loss.loc[f"{epoch}", "learning_rate"] = optimizer.param_groups[0]["lr"]

            if epoch < warmup_epochs:
                warmup_scheduler.step()
            else:
                plateau_scheduler.step(validation_loss)
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_corr = test_corr
            elif optimizer.param_groups[0]["lr"] <= min_lr and epoch > warmup_epochs:
                print(f"[{models_name} | num_layer={num_layer} | max_lr={format_float_no_sci_no_trailzero(max_lr)} | seed={random_seed} | fold={k_fold_index}] " f"Stopping at epoch {epoch} due to no improvement in validation loss.", flush=True)
                break

        save_csv_no_sci_append(path=f"{file}/k_fold_index-{k_fold_index}_loss.csv", new_df=loss.reset_index().rename(columns={"index": "epoch"}), append=False)
        k_fold_test_corr.append(best_corr)
    return float(pd.Series(k_fold_test_corr).mean()) - float(pd.Series(k_fold_test_corr).std())


@click.command()
@click.option("--random_seed", type=int, required=True)
def main(random_seed):
    model_number = config["cross_validation"]["model_number"]
    hyper_search = config["cross_validation"]["hyperparameter_search"]

    all_models = sorted(list(config["all_model"].keys()))
    model_combinations = [",".join(combo) for combo in itertools.combinations(all_models, model_number)]
    num_layer_list = list(range(hyper_search["num_layer"]["min"], hyper_search["num_layer"]["max"] + 1))
    max_lr_list = hyper_search["max_lr"]["choices"]

    full_grid = []
    for mc in model_combinations:
        for nl in num_layer_list:
            for lr in max_lr_list:
                full_grid.append({"model_combination": mc, "num_layer": nl, "max_lr": lr})

    filtered_grid = []
    for combo in full_grid:
        mc = combo["model_combination"]
        nl = combo["num_layer"]
        lr = combo["max_lr"]
        mc_dir = os.path.join("results", mc)
        this_run_dir = os.path.join(mc_dir, f"num_layer_{nl}_max_lr_{lr}_random_seed_{random_seed}")
        if not os.path.isdir(this_run_dir):
            filtered_grid.append(combo)

    result_dir = "best_score_results"
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/random_seed_{random_seed}.csv"

    if os.path.exists(result_path):
        existing_df = pd.read_csv(result_path)
        if "number" in existing_df.columns and len(existing_df) > 0:
            next_number_base = int(existing_df["number"].max()) + 1
        else:
            next_number_base = 0
    else:
        next_number_base = 0

    study_name = f"no-name-{uuid.uuid4()}"
    print(f"A new study created in memory with name: {study_name}", flush=True)

    best_value = None
    best_trial_number = None

    for i, combo in enumerate(filtered_grid):
        mc = combo["model_combination"]
        nl = combo["num_layer"]
        lr = combo["max_lr"]

        class DummyTrial:
            def __init__(self, mc_val, nl_val, lr_val):
                self._mc_val = mc_val
                self._nl_val = nl_val
                self._lr_val = lr_val

            def suggest_categorical(self, name, choices):
                if name == "model_combination":
                    return self._mc_val
                if name == "max_lr":
                    return self._lr_val
                raise ValueError(f"Unexpected categorical param: {name}")

            def suggest_int(self, name, low, high):
                if name == "num_layer":
                    return self._nl_val
                raise ValueError(f"Unexpected int param: {name}")

        score_val = objective(DummyTrial(mc, nl, lr), random_seed)
        trial_number = next_number_base + i

        if (best_value is None) or (score_val > best_value):
            best_value = score_val
            best_trial_number = trial_number
        params_str = f"{{'model_combination': '{mc}', 'num_layer': {nl}, 'max_lr': {format_float_no_sci_no_trailzero(lr)}}}"
        print(f"Trial {trial_number} finished with value: {format_float_no_sci_no_trailzero(score_val)} and parameters: {params_str}. " f"Best is trial {best_trial_number} with value: {format_float_no_sci_no_trailzero(best_value)}.", flush=True)

        row_df = pd.DataFrame([{"number": trial_number, "value": score_val, "params_max_lr": lr, "params_model_combination": mc, "params_num_layer": nl, "state": "COMPLETE"}], columns=["number", "value", "params_max_lr", "params_model_combination", "params_num_layer", "state"])
        save_csv_no_sci_append(path=result_path, new_df=row_df, append=True, dedup_cols=["number"])


if __name__ == "__main__":
    main()
