/* Koko in the browser, shown on the cartridge's own screen. koko.wasm is the cartridge's engine and screen code
 * (engine/koko.c, ui.c ...) compiled to WebAssembly; koko.bin is the cartridge's data. This file feeds it keys. */
'use strict';
const $ = id => document.getElementById(id);
const TOK_END = 0, TOK_USER = 1, TOK_KOKO = 2;
const INV_TEMP = 512, REPLY_MAX = 120, MSG_MAX = 56;         // as on the cartridge
const GBA_MS = 290;                                           // one token on the cartridge: about 3.4 per second
const NEW = '\u0001';                                         // the cartridge keyboard's "new chat" key
const HELP = '\u0002';                                        // the "helpers on/off" key: fact table and calculator
const ROWS = ['abcdefghijklm', 'nopqrstuvwxyz', '0123456789+-' + HELP, '.,!?\'___/()' + NEW + '<'];   // ___: the space bar, three cells wide
const QUICK = ['hi', 'who are you', 'tell me a story about a fox', 'what is the capital of France?', 'can you tell me a joke'];
const KEY_X = 3, KEY_Y = 99, KEY_W = 18, KEY_H = 15, CHIP_Y = 81, PILL_Y = 62;

let k, mem, ctx, image;
let msg = '', kx = 0, ky = 0, upper = 0, quick = -1;
let fed = [], userFed = false, nsug = 0;
let helpers = true, note = '';                                // note: a header message until the next change
let confirmNew = false;                                       // the new-chat key was pressed once
let view = 'title', stopAsked = false, blink = null;          // title |                       // typing | reply (Koko is writing) | done (reply on screen)

/* ---------- engine helpers ---------- */
const bytes = (p, n) => new Uint8Array(mem.buffer, p, n);
function putText(s, at = 0) { const b = bytes(k.web_text() + at, s.length); for (let i = 0; i < s.length; i++) b[i] = s.charCodeAt(i); }
function encode(s) { if (!s.length) return []; putText(s); const n = k.web_encode(s.length); return Array.from(new Uint16Array(mem.buffer, k.web_ids(), n)); }
function tokenText(id) { const n = k.web_token(id); return String.fromCharCode(...bytes(k.web_text(), n)); }
const isPrefix = (a, b) => a.length <= b.length && a.every((v, i) => v === b[i]);
function header(s) { putText(s, 512); k.web_ui_header(s.length); }

/* The model reads the message while it is typed, exactly as the cartridge does: finished words are fed at once;
 * an edit to a word already read rewinds to the start of the message. */
function readAlong() {
  const cut = msg.lastIndexOf(' ');
  const want = encode(cut > 0 ? msg.slice(0, cut) : '');
  if (!isPrefix(fed, want)) { k.web_rewind(); fed = []; userFed = false; }
  if (!userFed) { k.web_step(TOK_USER); userFed = true; }
  for (let i = fed.length; i < want.length; i++) { k.web_step(want[i]); fed.push(want[i]); }
  const prefix = msg.slice(cut + 1);
  if (/^[A-Za-z]*$/.test(prefix)) { putText(prefix, 512); nsug = k.web_suggest(prefix.length, cut < 0 ? 1 : 0); }
  else nsug = k.web_suggest(-1, 0);
}
function suggestion(i) { return tokenText(new Uint16Array(mem.buffer, k.web_ids(), 2)[i]).trim(); }

/* ---------- screen ---------- */
function present() {
  const fb = bytes(k.web_fb(), 240 * 160), pal = new Uint16Array(mem.buffer, k.web_pal(), 64), px = image.data;
  for (let i = 0, o = 0; i < fb.length; i++, o += 4) {
    const c = pal[fb[i]], r = c & 31, g = (c >> 5) & 31, b = (c >> 10) & 31;
    px[o] = (r << 3) | (r >> 2); px[o + 1] = (g << 3) | (g >> 2); px[o + 2] = (b << 3) | (b >> 2); px[o + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}
function drawTyping() {
  readAlong();
  putText(msg); k.web_ui_typing(msg.length, kx, ky, upper);
  k.web_ui_chips();
  header(note || (msg ? 'START send' : 'START ideas')); note = '';
  present();
}
function fit() {                                              // whole-number scale when there is room for 2x or more
  const pad = 20, panel = document.querySelector('.panel').offsetHeight + 40;
  const s = Math.min((innerWidth - 32 - pad) / 240, (innerHeight - panel - pad) / 160);
  const scale = s >= 2 ? Math.floor(s) : Math.max(1, s);
  $('lcd').style.width = Math.round(240 * scale) + 'px'; $('lcd').style.height = Math.round(160 * scale) + 'px';
}

/* ---------- input: the cartridge's buttons, driven by a real keyboard or a pointer ---------- */
function moveTo(ch) {                                          // the highlighted key follows what was typed
  const low = ch === ' ' ? '_' : ch.toLowerCase();
  for (let r = 0; r < 4; r++) { const c = ROWS[r].indexOf(low); if (c >= 0) { kx = low === '_' ? 6 : c; ky = r; } }
  if (/[A-Za-z]/.test(ch)) upper = ch !== low ? 1 : 0;
}
function typeChar(ch) {
  if (!(ch === ' ' || ROWS.join('').replace(/[_<\u0001\u0002]/g, '').includes(ch.toLowerCase()))) return;    // only what the cartridge keyboard has
  confirmNew = false;
  moveTo(ch);
  if (msg.length < MSG_MAX && !(ch === ' ' && (!msg || msg.endsWith(' ')))) msg += ch;
  drawTyping();
}
function del() { confirmNew = false; kx = 12; ky = 3; msg = msg.slice(0, -1); drawTyping(); }
function newKey() { newChat(); kx = ROWS[3].indexOf(NEW); ky = 3; drawTyping(); }      // one press, as on the cartridge
function setHelpers(on) {
  if (view === 'title') { helpers = on; k.web_ui_helpers(on ? 1 : 0); $('helpers').checked = on; return; }
  helpers = on; k.web_ui_helpers(on ? 1 : 0); $('helpers').checked = on;
  if (view === 'typing') { note = on ? 'helpers on' : 'model only'; kx = 12; ky = 2; drawTyping(); }
}
function accept(i) {
  if (i >= nsug) return;
  confirmNew = false;
  msg = (msg.slice(0, msg.lastIndexOf(' ') + 1) + suggestion(i) + ' ').slice(0, MSG_MAX);
  drawTyping();
}
function start() {
  confirmNew = false;
  if (msg) return void send();
  quick = (quick + 1) % QUICK.length; msg = QUICK[quick];      // START on an empty line: ready-made messages
  drawTyping();
}
function showTitle() {
  view = 'title'; k.web_ui_init(); k.web_ui_title(); present();
  let on = true, last = 0;
  blink = setInterval(() => { if (view !== 'title') return; on = !on; k.web_ui_title_prompt(on ? 1 : 0); }, 530);
  const snow = t => {                                          // the cartridge moves the snow once per frame, 60 times a second
    if (view !== 'title') return;
    if (t - last >= 16) { last = t; k.web_ui_title_tick(); present(); }
    requestAnimationFrame(snow);
  };
  requestAnimationFrame(snow);
}
function leaveTitle() { clearInterval(blink); newChat(); }
function wake() { if (view === 'done') { view = 'typing'; drawTyping(); } }
document.addEventListener('keydown', e => {
  if (!k || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.target instanceof HTMLButtonElement || e.target instanceof HTMLInputElement) { if (e.key === 'Enter' || e.key === ' ') return; }
  if (view === 'title') { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); leaveTitle(); } return; }
  if (view === 'reply') { if (e.key === 'Escape' || e.key === 'Backspace') { stopAsked = true; e.preventDefault(); } return; }
  const typed = e.key.length === 1, known = typed || ['Enter', 'Tab', 'Backspace'].includes(e.key);
  if (!known) return;
  e.preventDefault();
  wake();
  if (e.key === 'Enter') start();
  else if (e.key === 'Tab') accept(e.shiftKey ? 1 : 0);
  else if (e.key === 'Backspace') del();
  else typeChar(e.key);
});
$('lcd').addEventListener('pointerdown', e => {
  if (!k) return;
  const r = $('lcd').getBoundingClientRect();
  const x = Math.floor((e.clientX - r.left) * 240 / r.width), y = Math.floor((e.clientY - r.top) * 160 / r.height);
  if (view === 'title') return void leaveTitle();
  if (view === 'reply') { if (y >= 141 && y < 157 && x >= 210) stopAsked = true; return; }   // the stop square
  if (view === 'done') return void wake();
  if (y >= KEY_Y && x >= KEY_X && x < KEY_X + 13 * KEY_W) {
    const row = Math.min(3, Math.floor((y - KEY_Y) / KEY_H)), col = Math.floor((x - KEY_X) / KEY_W), ch = ROWS[row][col];
    if (ch === NEW) newKey(); else if (ch === HELP) setHelpers(!helpers); else if (ch === '<') del(); else typeChar(ch === '_' ? ' ' : (upper && row < 2 ? ch.toUpperCase() : ch));
  } else if (y >= CHIP_Y && y < CHIP_Y + 14) { const i = k.web_ui_chip_at(x); if (i >= 0) accept(i); }
  else if (y >= PILL_Y && y < PILL_Y + 16 && x >= 210) start();
});

/* ---------- one exchange, as on the cartridge ---------- */
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function send() {
  msg = msg.replace(/ +$/, '');
  if (!msg || view !== 'typing') return;
  view = 'reply'; stopAsked = false;
  const full = encode(msg);
  putText(msg); k.web_ui_reply_layout(msg.length);
  k.web_ui_reply_button(1);                                    // a stop square while Koko is answering
  header('reading'); present();
  if (!isPrefix(fed, full)) { k.web_rewind(); fed = []; userFed = false; }
  if (!userFed) k.web_step(TOK_USER);
  for (let i = fed.length; i < full.length; i++) k.web_step(full[i]);
  k.web_step(TOK_KOKO);
  putText(msg);
  const al = helpers ? k.web_sum(msg.length) : 0;              // a sum is worked out by the game
  let scripted = al ? encode(String.fromCharCode(...bytes(k.web_text() + 512, al))) : null;
  let mark = al ? 2 : 0;
  if (!scripted && helpers) { putText(msg); const n = k.web_fact(msg.length); if (n) mark = 1; if (n) scripted = Array.from(new Uint16Array(mem.buffer, k.web_ids(), n)); }   // a fact on file
  header('B stop');
  k.web_reply_begin(); k.web_ui_reply_start(mark); present();
  const gba = $('pace').checked, t0 = performance.now();
  let made = 0;
  if (gba) await sleep(GBA_MS);
  while (made < REPLY_MAX && !stopAsked) {
    const t = scripted ? (made < scripted.length ? scripted[made] : TOK_END) : k.web_pick(INV_TEMP);
    if (t === TOK_END || t === TOK_USER || t === TOK_KOKO) break;
    k.web_ui_reply_token(t); k.web_step(t); made++;
    present();
    if (gba) await sleep(GBA_MS); else if (made % 8 === 0) await sleep(0);
  }
  k.web_reply_end();
  k.web_ui_reply_button(0);
  k.web_ui_rate(made, Math.max(1, Math.round((performance.now() - t0) * 16.384)));
  present();
  k.web_mark(); fed = []; userFed = false; msg = '';
  view = 'done';                                               // the reply stays up until the next key
}
function newChat() {
  if (view === 'reply') return;
  clearInterval(blink);
  k.web_reset(); k.web_mark(); fed = []; userFed = false; msg = ''; kx = ky = upper = 0; quick = -1; confirmNew = false;
  k.web_ui_init(); k.web_ui_clear(); k.web_ui_helpers(helpers ? 1 : 0);
  const hello = "Woof! I'm Koko. Say something!";
  putText(hello); k.web_ui_say(0, hello.length);
  view = 'typing'; drawTyping(); $('lcd').focus();
}
$('new').addEventListener('click', newChat);
$('pace').addEventListener('change', () => $('lcd').focus());
$('helpers').addEventListener('change', e => { setHelpers(e.target.checked); $('lcd').focus(); });
addEventListener('resize', fit);

/* ---------- start-up ---------- */
async function fetchModel() {
  const r = await fetch('koko.bin');
  if (!r.ok) throw new Error(`koko.bin: ${r.status}`);
  const total = +r.headers.get('content-length') || 33300000;
  const reader = r.body.getReader(); const parts = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length;
    $('loadbar').style.width = Math.min(100, got / total * 100) + '%';
  }
  const all = new Uint8Array(got); let o = 0;
  for (const p of parts) { all.set(p, o); o += p.length; }
  return all;
}
(async () => {
  fit();
  try {
    const [wasm, blob] = await Promise.all([fetch('koko.wasm').then(r => r.arrayBuffer()), fetchModel()]);
    const { instance } = await WebAssembly.instantiate(wasm, {});
    k = instance.exports; mem = k.memory;
    const p = k.web_alloc(blob.length);
    if (!p) throw new Error('not enough memory');
    bytes(p, blob.length).set(blob);
    k.web_init(p);
    const q = new URLSearchParams(location.search).get('seed');       // ?seed=N makes replies repeatable
    k.web_seed(q ? +q : (crypto.getRandomValues(new Uint32Array(1))[0] || 1));
    ctx = $('lcd').getContext('2d'); image = ctx.createImageData(240, 160);
    $('load').hidden = true;
    showTitle(); $('lcd').focus();
    window.koko = { ready: true, state: () => ({ view, msg, nsug, helpers, s0: nsug > 0 ? suggestion(0) : '', s1: nsug > 1 ? suggestion(1) : '' }),
                    pixels: () => Array.from(bytes(k.web_fb(), 240 * 160)), palette: () => Array.from(new Uint16Array(mem.buffer, k.web_pal(), 64)) };
  } catch (e) { $('loadmsg').textContent = 'Koko could not start: ' + e.message; }
})();
