import os
import json
import torch
import importlib
from Bio import SeqIO
from tqdm import tqdm


def process_single_model(model_name, model_dir, saved_folder, config):
    module = importlib.import_module(f"{model_dir}.function")
    func_name = f"generate_{model_dir}"
    model_func = getattr(module, func_name)

    for i, mut_info in enumerate(tqdm(os.listdir(saved_folder), desc=model_name)):
        seq = str(list(SeqIO.parse(f"{saved_folder}/{mut_info}/result.fasta", "fasta"))[0].seq)
        embedding = model_func(seq)
        if i == 0:
            config["all_model"][model_name] = {"shape": list(embedding.shape)[-1]}
        torch.save(embedding, f"{saved_folder}/{mut_info}/{model_name}_embedding.pt")
        del embedding

    del model_func, module


def main():
    saved_folder = "../features"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with open("../config/config.json", "r") as f:
        config = json.load(f)

    if "all_model" not in config:
        config["all_model"] = {}

    config["all_model"].clear()

    model_dirs = [(item.replace("_embedding", ""), item) for item in os.listdir(".") if os.path.isdir(item) and item.endswith("_embedding")]
    print(f"Found models: {[name for name, _ in model_dirs]}")

    for model_name, model_dir in model_dirs:
        process_single_model(model_name, model_dir, saved_folder, config)

    with open("../config/config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
