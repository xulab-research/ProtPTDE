import torch
from transformers import AutoModelForMaskedLM

_esmc_300m_model_cache = None


def get_esmc_300m_model():
    global _esmc_300m_model_cache

    if _esmc_300m_model_cache is None:

        esmc_model = AutoModelForMaskedLM.from_pretrained("Synthyra/ESMplusplus_small", torch_dtype=torch.float32, trust_remote_code=True).eval().cuda()
        tokenizer = esmc_model.tokenizer

        _esmc_300m_model_cache = (esmc_model, tokenizer)

    return _esmc_300m_model_cache


def generate_esmc_300m_embedding(seq):
    with torch.no_grad():

        esmc_model, tokenizer = get_esmc_300m_model()

        tokenized = tokenizer([seq], padding=True, return_tensors="pt")
        tokenized = {key: value.cuda() for key, value in tokenized.items()}
        embedding = esmc_model(**tokenized).last_hidden_state[0, 1:-1, :].detach().cpu().clone()

        return embedding
