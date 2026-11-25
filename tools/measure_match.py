"""kokoSD 2.4: how many questions in other wordings find their article: the 2.3 matcher (git) on the 2.3 fact file
against the 2.4 matcher on the 2.4 file, with the answer of each so wrong matches show. GPL-3.0-or-later.
    python3 tools/measure_match.py OLD_FACTS.dat NEW_FACTS.dat [old git revision]"""
import subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
B = R / 'build'
def tool(src, name):
    (B / f'{name}.c').write_text(src.replace('#ifdef __arm__', '#if 1'))
    subprocess.run(['gcc', '-O2', '-w', f'-I{R}/engine', f'-I{R}/gen', '-o', str(B / name), str(R / 'host/facts_cover.c'),
                    str(B / f'{name}.c'), str(R / 'engine/koko.c'), str(R / 'engine/koko_tok.c')], check=True)
old_rev = sys.argv[3] if len(sys.argv) > 3 else 'a0e93a4'
tool(subprocess.run(['git', 'show', f'{old_rev}:engine/facts.c'], cwd=R, capture_output=True, text=True, check=True).stdout, 'facts_old')
tool((R / 'engine/facts.c').read_text(), 'facts_new')
Q = '''can you tell me about the moon
tell me something about volcanoes
have you heard of eminem
do you know mozart
do you know who albert einstein was
what does photosynthesis mean
info about the eiffel tower
i want to know about giraffes
what can you tell me about paris
who is the singer taylor swift
who was the composer mozart
what is the planet mars
who was the painter vincent van gogh
what is the city of london
tell me about the country japan
who was the famous scientist isaac newton
what is a volcano in iceland
who was napoleon from france
what is the tallest animal the giraffe
who is lionel messi the football player
what is mercury
what is the planet mercury
tell me about java
who is prince
what is python
what is a jaguar
tell me about the amazon river
what are dolphins
what are black holes
whats a tsunami
what's an earthquake
who's barack obama
tell me about dinosaurs please
what is chess?
explain gravity
explain to me what democracy is
what do you know about sharks
who invented the telephone
where is egypt
where is mount everest
when was world war ii
what is the capital of france
how big is the sun
what is the biggest ocean
can you tell me who was cleopatra
do you know anything about the roman empire
i like dogs
what is your name
tell me a story
how are you'''.split('\n')
def run(exe, facts):
    out = subprocess.run([str(B / exe), str(B / 'koko.bin'), facts, '1'], input='\n'.join(Q) + '\n',
                         capture_output=True, text=True, check=True).stdout.splitlines()
    return [l.split('\t', 1) for l in out]
a, b = run('facts_old', sys.argv[1]), run('facts_new', sys.argv[2])
na = sum(x[0] != 'none' for x in a); nb = sum(x[0] != 'none' for x in b)
for q, x, y in zip(Q, a, b):
    print(f'{q}\n   2.3: {x[1][:110] if x[0] != "none" else "-"}\n   2.4: {y[1][:110] if y[0] != "none" else "-"}')
print(f'answered: 2.3 {na}/{len(Q)}, 2.4 {nb}/{len(Q)}')
