"""How much Koko gains from the SD card's fact file (out/koko/facts.dat): the share of questions answered
from Wikipedia with the cartridge's own table only, and with the file. Two sets: 221 everyday questions
(animals, food, things, science, countries, cities, famous people, topics) and 2,000 article titles drawn at
random (seed 2026) from the file, asked as "tell me about <title>". Builds build/facts_cover (host/facts_cover.c
with engine/facts.c's SD path switched on). GPL-3.0-or-later."""
import random, struct, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
B = R / 'build'
src = (R / 'engine/facts.c').read_text().replace('#ifdef __arm__', '#if 1')
(B / 'facts_host.c').write_text(src)
subprocess.run(['gcc', '-O2', '-w', f'-I{R}/engine', f'-I{R}/gen', '-o', str(B / 'facts_cover'), str(R / 'host/facts_cover.c'),
                str(B / 'facts_host.c'), str(R / 'engine/koko.c'), str(R / 'engine/koko_tok.c')], check=True)
fd = (R / 'out/koko/facts.dat').read_bytes()
count, _, index_at, entry_at, _ = struct.unpack('<5I', fd[8:28])
offs = sorted(set(struct.unpack_from('<I', fd, index_at + 8 * i + 4)[0] for i in range(count)))
titles = [fd[entry_at + o + 7:entry_at + o + 7 + fd[entry_at + o + 6]].decode('utf-8', 'replace') for o in offs]
random.seed(2026)
rand = [f'tell me about {t}' for t in random.sample(titles, 2000)]
W = str.split
every = ([f'what is a {w}?' for w in W('dog cat horse cow pig sheep goat chicken duck rabbit mouse lion tiger elephant giraffe zebra '
          'bear wolf fox deer kangaroo koala penguin dolphin whale shark octopus eagle owl parrot snake frog turtle crocodile bee '
          'ant butterfly spider camel gorilla hippopotamus apple banana orange strawberry grape lemon potato tomato carrot onion '
          'rice bread cheese milk egg chocolate pizza pasta sushi coffee tea honey sugar salt butter car bicycle airplane train '
          'ship computer telephone television camera piano guitar violin drum clock book pencil umbrella bridge castle pyramid '
          'volcano mountain river ocean desert forest island rainbow snow rain thunderstorm earthquake')]
         + [f'what is {w}?' for w in W('sun moon earth mars jupiter saturn star galaxy planet comet atom molecule electricity '
            'magnet gravity photosynthesis dna cell bacteria virus oxygen water') + ['carbon dioxide', 'iron', 'gold', 'diamond']]
         + [f'tell me about {w}' for w in W('france germany japan china india brazil canada mexico italy spain egypt kenya australia '
            'russia sweden norway greece turkey argentina nigeria paris london berlin tokyo rome cairo moscow sydney madrid beijing '
            'istanbul amsterdam vienna') + ['new york city', 'los angeles']]
         + [f'who was {w}?' for w in ['albert einstein', 'isaac newton', 'charles darwin', 'marie curie', 'william shakespeare',
            'wolfgang amadeus mozart', 'ludwig van beethoven', 'leonardo da vinci', 'pablo picasso', 'napoleon', 'julius caesar',
            'cleopatra', 'abraham lincoln', 'george washington', 'martin luther king jr', 'nelson mandela', 'mahatma gandhi',
            'queen victoria', 'winston churchill', 'elvis presley', 'michael jackson', 'the beatles', 'walt disney', 'neil armstrong',
            'christopher columbus', 'galileo galilei', 'nikola tesla', 'thomas edison', 'vincent van gogh', 'frida kahlo']]
         + [f'what is {w}?' for w in ['football', 'basketball', 'tennis', 'chess', 'olympic games', 'music', 'painting', 'poetry',
            'democracy', 'money', 'school', 'internet', 'video game', 'game boy advance', 'nintendo', 'robot', 'dinosaur',
            'world war ii', 'ancient egypt', 'roman empire', 'renaissance', 'christmas', 'halloween', 'birthday', 'friendship',
            'love', 'sleep', 'dream', 'language', 'mathematics', 'history', 'philosophy']])
def run(qs, mode):
    out = subprocess.run([str(B / 'facts_cover'), str(B / 'koko.bin'), str(R / 'out/koko/facts.dat'), str(mode)],
                         input='\n'.join(qs) + '\n', capture_output=True, text=True, check=True).stdout.splitlines()
    return [l.split('\t', 1)[0] for l in out]
for name, qs in (('everyday questions', every), ('random Wikipedia subjects', rand)):
    a, b = run(qs, 0), run(qs, 1)
    n, x, y = len(qs), sum(s != 'none' for s in a), sum(s != 'none' for s in b)
    print(f'{name}: {n}; cartridge table only {x} ({100 * x / n:.1f}%); with the SD file {y} ({100 * y / n:.1f}%)')
    print('  not answered:', [q for q, s in zip(qs, b) if s == 'none'][:12])
