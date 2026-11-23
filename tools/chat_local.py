import json, sys, numpy as np, torch
sys.path.insert(0, '/home/user/kokosd')
from tokenizers import Tokenizer
from train.model import Config, Model
import dataclasses
run = sys.argv[1]; A = '/home/user/kokosd/artifacts/'
res = json.load(open(f'{A}models/{run}/result.json'))
names = {f.name for f in dataclasses.fields(Config)}
cfg = Config(**{k: v for k, v in res['config'].items() if k in names})
tok = Tokenizer.from_file(A + 'mix1x/mix1/tok_16384.json')
model = Model(cfg, {"counts": torch.from_numpy(np.load(A + 'mix1x/mix1/counts_16384.npy'))})
print(model.load_state_dict(torch.load(f'{A}models/{run}/model.pt', map_location='cpu')))
model.set_progress(1.0); model.eval()
torch.manual_seed(0)
def reply(text, temp=0.7, topk=40, n=90):
    ids = tok.encode(text).ids
    out = []
    for _ in range(n):
        with torch.no_grad():
            lg = model(torch.tensor([ids[-cfg.T:]]))[0][0, -1] / temp
        v, i = lg.topk(topk); nxt = i[torch.multinomial(v.softmax(-1), 1)].item()
        ids.append(nxt); out.append(nxt)
        s = tok.decode(out, skip_special_tokens=False)
        if '<|user|>' in s or '<|' in s[-12:] and s.endswith('|>'): break
    return tok.decode(out, skip_special_tokens=False).split('<|user|>')[0].strip()
for q in sys.argv[2:]:
    print('\nUSER:', q); print('KOKO:', reply(f'<|user|>{q}<|koko|>'))
