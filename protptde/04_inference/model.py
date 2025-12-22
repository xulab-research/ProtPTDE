import json
import torch

with open("../config/config.json", "r", encoding="utf-8") as f:
    config = json.load(f)


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
