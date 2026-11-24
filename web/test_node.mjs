// The browser build must write exactly what the cartridge engine writes: same prompt, same seed, same tokens.
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
const dir = new URL('./site/', import.meta.url).pathname;
const { instance } = await WebAssembly.instantiate(readFileSync(dir + 'koko.wasm'), {});
const k = instance.exports;
const blob = readFileSync(dir + 'koko.bin');
const p = k.web_alloc(blob.length);
new Uint8Array(k.memory.buffer, p, blob.length).set(blob);
k.web_init(p);
const enc = s => { new Uint8Array(k.memory.buffer, k.web_text(), s.length).set([...s].map(c => c.charCodeAt(0))); const n = k.web_encode(s.length); return [...new Uint16Array(k.memory.buffer, k.web_ids(), n)]; };
const dec = id => { const n = k.web_token(id); return String.fromCharCode(...new Uint8Array(k.memory.buffer, k.web_text(), n)); };
let bad = 0;
for (const [q, seed] of [['hi', 7], ['who are you', 99], ['tell me a story about a fox', 1234], ['what is the capital of France?', 5]]) {
  const prompt = [1, ...enc(q), 2];
  k.web_reset(); k.web_seed(seed);
  for (const t of prompt) k.web_step(t);
  k.web_reply_begin();
  const out = []; const t0 = performance.now();
  for (let i = 0; i < 120; i++) { const t = k.web_pick(512); if (t < 3) break; out.push(t); k.web_step(t); }
  const ms = performance.now() - t0;
  k.web_reply_end();
  const native = execFileSync(new URL('../build/koko_gen', import.meta.url).pathname, [dir + 'koko.bin', String(seed), '512', '120', '1', '0', '1'], { input: prompt.join(' ') }).toString().trim().split(/\s+/).filter(Boolean).map(Number);
  const same = JSON.stringify(out) === JSON.stringify(native);
  bad += !same;
  console.log(same ? 'PASS' : 'FAIL', JSON.stringify(q), '->', JSON.stringify(out.map(dec).join('')), `(${out.length} tokens, ${(out.length / ms * 1000).toFixed(0)} tokens/s in Node)`);
}
process.exit(bad ? 1 : 0);
