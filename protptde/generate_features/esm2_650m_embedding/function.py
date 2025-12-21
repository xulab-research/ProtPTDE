import torch

_esm2_model_cache = None


def get_esm2_model():
    global _esm2_model_cache

    if _esm2_model_cache is None:

        esm2_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        batch_converter = alphabet.get_batch_converter()
        esm2_model = esm2_model.eval().cuda()

        _esm2_model_cache = (esm2_model, batch_converter)

    return _esm2_model_cache


def generate_esm2_650m_embedding(seq):
    with torch.no_grad():
        esm2_model, batch_converter = get_esm2_model()

        _, _, batch_tokens = batch_converter([("", seq)])
        embedding = esm2_model(batch_tokens.cuda(), repr_layers=[33], return_contacts=False)["representations"][33][0, 1:-1, :].detach().cpu().clone()

        return embedding
