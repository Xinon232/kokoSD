"""The models on the scoreboard and one way to call each: score text (nats) and continue a prompt."""
import os, sys, math, subprocess, json
import torch
sys.path.insert(0, '/home/user/kokosd/study'); sys.path.insert(0, '/home/user/kokosd')
os.environ.setdefault("HF_HOME", "/home/user/kokosd/scoreboard/hf")
torch.set_num_threads(4)
BASELINES = {                      # name -> (hub id, kind)
    "GPT-2 small": ("gpt2", "lm"),
    "SmolLM2-135M": ("HuggingFaceTB/SmolLM2-135M", "lm"),
    "SmolLM2-135M-Instruct": ("HuggingFaceTB/SmolLM2-135M-Instruct", "chat"),
    "TinyStories-33M": ("roneneldan/TinyStories-33M", "lm"),
    "TinyStories-8M": ("roneneldan/TinyStories-8M", "lm"),
    "TinyStories-1M": ("roneneldan/TinyStories-1M", "lm"),
}
_cache = {}


def hf(name):
    if name not in _cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        rid, kind = BASELINES[name]
        tok = AutoTokenizer.from_pretrained(rid)
        m = AutoModelForCausalLM.from_pretrained(rid, torch_dtype=torch.float32).eval()
        _cache[name] = (tok, m, kind)
    return _cache[name]


def hf_nats(name, text):
    """Total nats the model assigns to `text`, starting from its beginning-of-text token."""
    tok, m, _ = hf(name)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    ids = [bos] + tok.encode(text, add_special_tokens=False)
    ids = ids[:m.config.max_position_embeddings if hasattr(m.config, "max_position_embeddings") else 1024]
    x = torch.tensor([ids])
    with torch.no_grad():
        lg = m(x).logits[0, :-1]
    return float(torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum")), len(ids) - 1


def hf_generate(name, prompt, max_new=60, temp=0.0, seed=0, chat=False):
    tok, m, kind = hf(name)
    torch.manual_seed(seed)
    if chat and kind == "chat":
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
    else:
        ids = tok(prompt, return_tensors="pt")
    kw = dict(do_sample=True, temperature=temp, top_k=40) if temp > 0 else dict(do_sample=False)
    with torch.no_grad():
        out = m.generate(**ids, max_new_tokens=max_new, pad_token_id=tok.eos_token_id, repetition_penalty=1.0, **kw)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


# ---------------- Koko ----------------
def koko_tok():
    from common import tokenizer
    return tokenizer()


def koko_torch(run):
    if ("koko", run) not in _cache:
        from common import load
        _cache[("koko", run)] = load(run)[0]
    return _cache[("koko", run)]


def koko_nats(run, text):
    """Total nats for `text` after the document separator, with the training code's model (full output step)."""
    tok = koko_tok()
    ids = [0] + tok.encode(text).ids
    x = torch.tensor([ids])
    with torch.no_grad():
        lg = koko_torch(run)(x[:, :-1])[0][0]
    return float(torch.nn.functional.cross_entropy(lg, x[0, 1:], reduction="sum")), len(ids) - 1


G = '/home/user/kokosd/build/'
def koko_reply(history, seed=1, temp=0.5, maxn=120, helpers=False):
    """The cartridge engine (native build): reply to a chat. helpers=True adds the cartridge's calculator and fact table."""
    tok = koko_tok()
    last = history[-1][1]
    if helpers:
        s = subprocess.run([G + 'sums_test'], input=last + '\n', capture_output=True, text=True).stdout.strip()
        if s: return s
        f = subprocess.run([G + 'facts_test', G + 'koko.bin'], input=last + '\n', capture_output=True, text=True).stdout.strip()
        if f: return f
    ids = []
    for who, t in history: ids += [1 if who == "user" else 2] + tok.encode(t).ids
    ids += [2]
    out = subprocess.run([G + 'koko_gen', G + 'koko.bin', str(seed), str(round(256 / temp)), str(maxn), '1', '0', '1'],
                         input=' '.join(map(str, ids)).encode(), capture_output=True).stdout.split()
    return tok.decode([int(x) for x in out], skip_special_tokens=False).strip()
