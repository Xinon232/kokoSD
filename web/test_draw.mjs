// The chat is drawn incrementally (rows slide up, only changed lines are redrawn). After every token of several long
// replies, the screen must equal what a full redraw gives.
import { readFileSync } from 'node:fs';
const dir = new URL('./site/', import.meta.url).pathname;
const { instance } = await WebAssembly.instantiate(readFileSync(dir + 'koko.wasm'), {});
const k = instance.exports;
const blob = readFileSync(dir + 'koko.bin');
const p = k.web_alloc(blob.length);
new Uint8Array(k.memory.buffer, p, blob.length).set(blob);
k.web_init(p); k.web_seed(5);
const put = (s, at = 0) => new Uint8Array(k.memory.buffer, k.web_text() + at, s.length).set([...s].map(c => c.charCodeAt(0)));
const enc = s => { put(s); const n = k.web_encode(s.length); return [...new Uint16Array(k.memory.buffer, k.web_ids(), n)]; };
const fb = () => Uint8Array.from(new Uint8Array(k.memory.buffer, k.web_fb(), 240 * 160));
k.web_ui_init(); k.web_ui_clear(); put("Woof! I'm Koko. Say something!"); k.web_ui_say(0, 30);
let checks = 0, bad = 0;
for (const [msg, mark] of [['tell me a story about a fox', 0], ['who was mozart?', 1], ['tell me a long story about a dragon and a knight', 0], ['hi', 0], ['what is 17 + 25?', 2], ['tell me a story about a robot', 0]]) {
  const ids = enc(msg);
  put(msg); k.web_ui_reply_layout(msg.length); k.web_ui_reply_button(1);
  k.web_step(1); for (const t of ids) k.web_step(t); k.web_step(2);
  k.web_reply_begin(); k.web_ui_reply_start(mark);
  for (let i = 0; i < 120; i++) {
    const t = k.web_pick(512); if (t < 3) break;
    k.web_ui_reply_token(t); k.web_step(t);
    const a = fb(); k.web_ui_redraw(); const b = fb();
    checks++;
    if (!a.every((v, n) => v === b[n])) { bad++; if (bad < 3) console.log('differs after token', i, 'of', JSON.stringify(msg)); }
  }
  k.web_reply_end();
}
console.log(bad ? 'FAIL' : 'PASS', `incremental drawing equals a full redraw at ${checks} of ${checks - bad + bad} steps` + (bad ? ` (${bad} differ)` : ''));
process.exit(bad ? 1 : 0);
