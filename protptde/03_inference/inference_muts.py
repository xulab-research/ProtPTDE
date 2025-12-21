import sys
import json
import click
import torch
import importlib
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from pathlib import Path
from model import ModelUnion

sys.path.append(str(Path(__file__).parent.parent))

with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def generate_wt_input(seq):
    input_dict = {"seq": seq}
    with torch.no_grad():
        for model_name in selected_models:
            embedding = torch.load(f"../features/wt/{model_name}_embedding.pt").unsqueeze(0).cuda()
            input_dict[f"{model_name}_embedding"] = embedding
    return input_dict


def generate_mut_input(seq):
    input_dict = {"seq": seq}
    with torch.no_grad():
        for model_name in selected_models:
            embedding = embedding_functions[model_name](seq).unsqueeze(0).cuda()
            input_dict[f"{model_name}_embedding"] = embedding
    return input_dict


best_hyperparameters = config["best_hyperparameters"]
selected_models = best_hyperparameters["selected_models"]
num_layer = best_hyperparameters["num_layer"]

wt_seq = str(list(SeqIO.parse("../features/wt/result.fasta", "fasta"))[0].seq)

embedding_functions = {}
for model_name in selected_models:
    module = importlib.import_module(f"generate_features.{model_name}_embedding.function")
    func_name = f"generate_{model_name}_embedding"
    embedding_functions[model_name] = getattr(module, func_name)


@click.command()
@click.option("--mut_counts", required=True, type=int)
def main(mut_counts):
    data = pd.read_csv(f"sorted_mut_counts_{mut_counts}.csv", index_col=0)
    model = ModelUnion(num_layer, selected_models)

    model_state_dict = model.state_dict().copy()
    model_state_dict.update(torch.load("../02_final_model/finetune_best.pth").copy())
    model.load_state_dict(model_state_dict)

    model.eval().cuda()

    wt_data = generate_wt_input(wt_seq)

    with torch.no_grad():
        for mut_name in tqdm(data.index):
            mut_seq = data.loc[mut_name, "mut_seq"]
            mut_data = generate_mut_input(mut_seq)
            data.loc[mut_name, "pred"] = model(wt_data, mut_data).item()

    for multi_mut_info in data.index:
        tmp_str = ""
        for single_mut_info in multi_mut_info.split(","):
            tmp_str += single_mut_info[0] + str(int(single_mut_info[1:-1]) + 1) + single_mut_info[-1] + ","
        data.loc[multi_mut_info, "mut_name"] = tmp_str[:-1]

    data = data.set_index("mut_name")
    data = data[["pred"]]
    data = data.sort_values("pred", ascending=False)
    data.to_csv(f"predicted_sorted_mut_counts_{mut_counts}.csv")


if __name__ == "__main__":
    main()
