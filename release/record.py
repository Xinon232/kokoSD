"""Record a short session in mGBA as release/screenshots/koko.gif and refresh the store screenshots."""
import os as _os; ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))  # repository root
import subprocess, os, shutil
os.chdir(ROOT)
exec(open('engine/test_game.py').read().split("ok = True")[0])
from PIL import Image
os.makedirs('out/gif', exist_ok=True)
n = [0]
def shot(): n[0] += 1; return f'shot:out/gif/f{n[0]:03d}.ppm'
steps = ['f:20', shot(), 'f:32', shot(), 'f:32', shot(), f'k:{START}', 'f:12', shot()]
pad = Pad()
for ch in 'tell me a ':
    steps += pad.key(ch) + ['f:4', shot()]
steps += [IDLE, 'f:6', shot(), f'k:{R}', IDLE, 'f:6', shot()]
for ch in 'about a fox':
    steps += pad.key(ch) + ['f:3', shot()]
steps += [f'k:{START}']
for i in range(46): steps += ['f:22', shot()]
steps += [f'until:{DBG + 12:#x}:1:8000', 'f:5', shot(), 'shot:out/shot_story.ppm', 'f:40', shot()]
for ch in 'who was mozart?':
    steps += pad.key(ch) + ['f:3', shot()]
steps += [f'k:{START}']
for i in range(22): steps += ['f:22', shot()]
steps += [f'until:{DBG + 12:#x}:2:8000', 'f:5', shot(), 'f:60', shot()]
subprocess.run(['build/runner', ROM] + steps, check=True)
frames = [Image.open(f'out/gif/f{i:03d}.ppm').convert('RGB').resize((480, 320), Image.NEAREST) for i in range(1, n[0] + 1)]
dur = [900, 500, 500, 500] + [220] * (len(frames) - 4); dur[-1] = 3000
fp = [f.quantize(64, method=Image.MEDIANCUT) for f in frames]
fp[0].save('release/screenshots/koko.gif', save_all=True, append_images=fp[1:], duration=dur, loop=0, optimize=True)
big = lambda src, dst: Image.open(src).convert('RGB').resize((720, 480), Image.NEAREST).save(dst)
big('out/koko_title.ppm', 'release/screenshots/1-title.png'); big('out/shot_story.ppm', 'release/screenshots/2-story.png')
big('out/koko_suggest.ppm', 'release/screenshots/3-suggestions.png'); big('out/koko_fact.ppm', 'release/screenshots/4-fact-table.png')
big('out/koko_sum.ppm', 'release/screenshots/5-sum.png')
print(n[0], 'frames;', os.path.getsize('release/screenshots/koko.gif') // 1024, 'KB')
