"""Shared loading for post-training studies of the final models (CPU)."""
import dataclasses, json, sys
import numpy as np, torch
sys.path.insert(0, '/home/user/kokosd')
from tokenizers import Tokenizer
from train.model import Config, Model

A = '/home/user/kokosd/artifacts/'
DOMAINS = ("koko", "chat", "facts", "stories")
REF = {"koko": 0.4, "chat": 0.2, "facts": 0.2, "stories": 0.2}


def load(run="h_koko_L8"):
    res = json.load(open(f'{A}models/{run}/result.json'))
    names = {f.name for f in dataclasses.fields(Config)}
    cfg = Config(**{k: v for k, v in res['config'].items() if k in names})
    counts = torch.from_numpy(np.load(A + 'mix1x/mix1/counts_16384.npy'))
    model = Model(cfg, {"counts": counts})
    model.load_state_dict(torch.load(f'{A}models/{run}/model.pt', map_location='cpu'))
    model.set_progress(1.0)
    model.eval()
    return model, cfg, counts, res


def tokenizer():
    return Tokenizer.from_file(A + 'mix1x/mix1/tok_16384.json')


def val_seqs(domain, n, T, seed=0):
    """n windows of T+1 tokens from a domain's validation stream, evenly spaced."""
    ids = np.fromfile(f'{A}mix1x/mix1/{domain}_val_16384.bin', dtype=np.uint16).astype(np.int64)
    starts = np.linspace(0, len(ids) - T - 2, n).astype(int)
    return torch.from_numpy(np.stack([ids[s:s + T + 1] for s in starts]))
