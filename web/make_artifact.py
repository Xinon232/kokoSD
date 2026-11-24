"""web/site -> web/artifact: one HTML file (styles and script inline) plus koko.wasm and the model data as three
data-only WebAssembly modules (the page host serves .wasm but not raw .bin files, and caps files at 15 MB)."""
import os, shutil
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def leb(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n: return bytes(out)
def sec(i, body): return bytes([i]) + leb(len(body)) + body
def data_module(d):
    return (b'\x00asm\x01\x00\x00\x00' + sec(5, leb(1) + b'\x00' + leb((len(d) + 65535) >> 16))
            + sec(7, leb(1) + leb(1) + b'm' + b'\x02' + leb(0)) + sec(11, leb(1) + b'\x00' + b'\x41\x00\x0b' + leb(len(d)) + d))
css, js, html = (open('site/' + f).read() for f in ('style.css', 'app.js', 'index.html'))
body = html[html.index('<main class="app">'):html.index('<script src="app.js">')]
data = open('site/koko.bin', 'rb').read(); n = (len(data) + 2) // 3
os.makedirs('artifact', exist_ok=True)
for i in range(3):
    open(f'artifact/koko-data-{i + 1}.wasm', 'wb').write(data_module(data[i * n:(i + 1) * n]))
a, b = js.index('async function fetchModel() {'), js.index('(async () => {')
js = js[:a] + f'''/* The model data is published as three WebAssembly modules that hold nothing but bytes (exported memory "m"). */
const PARTS = [['koko-data-1.wasm', {n}], ['koko-data-2.wasm', {n}], ['koko-data-3.wasm', {len(data) - 2 * n}]], TOTAL = {len(data)};
async function fetchModel() {{
  const all = new Uint8Array(TOTAL); let got = 0;
  for (const [url, size] of PARTS) {{
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${{url}}: ${{r.status}}`);
    const chunks = []; let have = 0;
    const reader = r.body.getReader();
    for (;;) {{
      const {{ done, value }} = await reader.read();
      if (done) break;
      chunks.push(value); have += value.length;
      $('loadbar').style.width = Math.min(100, (got + Math.min(have, size)) / TOTAL * 100) + '%';
    }}
    const buf = new Uint8Array(have); let o = 0;
    for (const c of chunks) {{ buf.set(c, o); o += c.length; }}
    const {{ instance }} = await WebAssembly.instantiate(buf.buffer, {{}});
    all.set(new Uint8Array(instance.exports.m.buffer, 0, size), got);
    got += size;
  }}
  return all;
}}
''' + js[b:]
seed = "    const q = new URLSearchParams(location.search).get('seed');       // ?seed=N makes replies repeatable\n    k.web_seed(q ? +q : (crypto.getRandomValues(new Uint32Array(1))[0] || 1));"
assert seed in js
js = js.replace(seed, "    k.web_seed(crypto.getRandomValues(new Uint32Array(1))[0] || 1);")
page = f'''<title>Koko</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap">
<style>
{css}</style>
{body}<script>
{js}</script>
'''
assert '<html' not in page and '<body' not in page
open('artifact/koko.html', 'w').write(page)
shutil.copy('site/koko.wasm', 'artifact/koko.wasm')
print('artifact/koko.html', len(page), 'bytes')
