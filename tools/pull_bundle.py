"""Pull a bundle directory of split parts from the machine, verify every part and every joined file by SHA-256.
usage: pull.py <remote bundle dir under workspace/koko/bundle> <local dir>"""
import hashlib, os, sys
from concurrent.futures import ThreadPoolExecutor
sys.argv, argv = ['j'], sys.argv
exec(open(os.path.expanduser('~/.koko/j.py')).read().split('if __name__')[0])
remote, local = 'workspace/koko/bundle/' + argv[1], argv[2]
os.makedirs(local + '/parts', exist_ok=True)
sums = dict(reversed(l.split()) for l in get(remote + '/SHA256SUMS').decode().splitlines() if l.strip())
parts = dict(reversed(l.split()) for l in get(remote + '/PARTSUMS').decode().splitlines() if l.strip())
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
def one(name):
    p = f'{local}/parts/{name}'
    for attempt in range(5):
        if os.path.exists(p) and sha(p) == parts[name]: return name, True
        try: get(f'{remote}/{name}', p)
        except Exception as e: print('retry', name, type(e).__name__, flush=True)
    return name, os.path.exists(p) and sha(p) == parts[name]
with ThreadPoolExecutor(6) as ex:
    bad = [n for n, ok in ex.map(one, sorted(parts)) if not ok]
if bad: sys.exit(f'FAILED parts: {bad}')
ok = True
for name, want in sums.items():
    with open(f'{local}/{name}', 'wb') as out:
        for pn in sorted(n for n in parts if n.startswith(name + '.part')):
            out.write(open(f'{local}/parts/{pn}', 'rb').read())
    got = sha(f'{local}/{name}')
    print(name, os.path.getsize(f'{local}/{name}'), 'OK' if got == want else 'MISMATCH', flush=True)
    ok &= got == want
if not ok: sys.exit('FAILED whole-file checksum')
for pn in parts: os.remove(f'{local}/parts/{pn}')
print('ALL VERIFIED', len(sums), 'files')
