import json
import torch
from Bio import SeqIO
import soft_rank_pytorch


with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


def spearman_loss(pred, true, regularization_strength, regularization):
    assert pred.device == true.device
    assert pred.shape == true.shape
    assert pred.shape[0] == 1
    assert pred.ndim == 2

    device = pred.device

    soft_pred = soft_rank_pytorch.soft_rank(pred.cpu(), regularization_strength=regularization_strength, regularization=regularization).to(device)
    soft_true = _rank_data(true.squeeze(0)).to(device)
    preds_diff = soft_pred - soft_pred.mean()
    target_diff = soft_true - soft_true.mean()

    cov = (preds_diff * target_diff).mean()
    preds_std = torch.sqrt((preds_diff * preds_diff).mean())
    target_std = torch.sqrt((target_diff * target_diff).mean())

    spearman_corr = cov / (preds_std * target_std + 1e-6)
    return -spearman_corr


def _find_repeats(data):
    temp = data.detach().clone()
    temp = temp.sort()[0]

    change = torch.cat([torch.tensor([True], device=temp.device), temp[1:] != temp[:-1]])
    unique = temp[change]
    change_idx = torch.cat([torch.nonzero(change), torch.tensor([[temp.numel()]], device=temp.device)]).flatten()
    freq = change_idx[1:] - change_idx[:-1]
    atleast2 = freq > 1
    return unique[atleast2]


def _rank_data(data):
    n = data.numel()
    rank = torch.empty_like(data)
    idx = data.argsort()
    rank[idx[:n]] = torch.arange(1, n + 1, dtype=data.dtype, device=data.device)

    repeats = _find_repeats(data)
    for r in repeats:
        condition = data == r
        rank[condition] = rank[condition].mean()
    return rank


def spearman_corr(pred, true):
    assert pred.dtype == true.dtype
    assert pred.ndim <= 2 and true.ndim <= 2

    if pred.ndim == 1:
        pred = _rank_data(pred)
        true = _rank_data(true)
    else:
        pred = torch.stack([_rank_data(p) for p in pred.T]).T
        true = torch.stack([_rank_data(t) for t in true.T]).T

    preds_diff = pred - pred.mean(0)
    target_diff = true - true.mean(0)

    cov = (preds_diff * target_diff).mean(0)
    preds_std = torch.sqrt((preds_diff * preds_diff).mean(0))
    target_std = torch.sqrt((target_diff * target_diff).mean(0))

    spearman_corr = cov / (preds_std * target_std + 1e-6)
    return torch.clamp(spearman_corr, -1.0, 1.0)


class BatchData(torch.utils.data.Dataset):

    def __init__(self, csv, selected_models):
        self.csv = csv
        self.selected_models = selected_models

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        mut_info = self.csv.iloc[index].name
        wt_data = {"seq": str(list(SeqIO.parse("../features/wt/result.fasta", "fasta"))[0].seq)}
        mut_data = {"seq": str(list(SeqIO.parse(f"../features/{mut_info}/result.fasta", "fasta"))[0].seq)}
        for model_name in self.selected_models:
            wt_data[f"{model_name}_embedding"] = torch.load(f"../features/wt/{model_name}_embedding.pt")
            mut_data[f"{model_name}_embedding"] = torch.load(f"../features/{mut_info}/{model_name}_embedding.pt")

        return wt_data, mut_data, torch.tensor(self.csv.loc[mut_info, "label"]).to(torch.float32)


def to_gpu(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.to(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [to_gpu(i, device=device) for i in obj]
    elif isinstance(obj, tuple):
        return (to_gpu(i, device=device) for i in obj)
    elif isinstance(obj, dict):
        return {i: to_gpu(j, device=device) for i, j in obj.items()}
    else:
        return obj


class DownStreamModel(torch.nn.Module):

    def __init__(self, num_layer, selected_models):

        super().__init__()

        self.config = config
        self.selected_models = selected_models
        self.model_transforms = torch.nn.ModuleDict()

        embedding_output_dim = self.config["single_model_embedding_output_dim"]
        for model_name in self.selected_models:
            input_dim = self.config["all_model"][model_name]["shape"]
            self.model_transforms[model_name] = torch.nn.Sequential(torch.nn.LayerNorm(input_dim), torch.nn.Linear(input_dim, embedding_output_dim), torch.nn.LeakyReLU())

        layers = []
        input_dim = len(self.selected_models) * embedding_output_dim
        for _ in range(num_layer):
            layers.append(torch.nn.Linear(input_dim, 64))
            layers.append(torch.nn.LeakyReLU())
            input_dim = 64

        layers.append(torch.nn.Linear(64, 1))
        self.read_out = torch.nn.Sequential(*layers)

    def forward(self, embeddings_dict):
        transformed_embeddings = []
        for model_name in self.selected_models:
            embedding = embeddings_dict[f"{model_name}_embedding"]
            transformed = self.model_transforms[model_name](embedding)
            transformed_embeddings.append(transformed)

        x = torch.cat(transformed_embeddings, dim=-1)
        x = self.read_out(x)
        return x


class ModelUnion(torch.nn.Module):

    def __init__(self, num_layer, selected_models):
        super().__init__()
        self.down_stream_model = DownStreamModel(num_layer, selected_models)
        self.finetune_coef = torch.nn.Parameter(torch.tensor([1.0], requires_grad=False))

    def forward(self, wt_data, mut_data):
        wt_embeddings = {key: emb for key, emb in wt_data.items() if key.endswith("_embedding")}
        mut_embeddings = {key: emb for key, emb in mut_data.items() if key.endswith("_embedding")}

        wt_value = self.down_stream_model(wt_embeddings)
        mut_value = self.down_stream_model(mut_embeddings)

        wt_seq, mut_seq = wt_data["seq"], mut_data["seq"]
        if not isinstance(wt_seq, list):
            wt_seq, mut_seq = [wt_seq], [mut_seq]
        device = mut_value.device
        mut_pos = torch.stack([torch.tensor([int(wt_aa != mut_aa) for wt_aa, mut_aa in zip(wt_item, mut_item)], dtype=torch.int, device=device) for wt_item, mut_item in zip(wt_seq, mut_seq)])

        delta_value = (mut_value - wt_value).squeeze(-1) * mut_pos
        delta_value = delta_value.sum(1)
        return self.finetune_coef * delta_value
