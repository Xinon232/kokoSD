// Drive the real page in headless Chromium, and compare its screen with the cartridge's screen from mGBA.
const { chromium } = require('playwright');
const { execFileSync, spawn } = require('node:child_process');
const fs = require('node:fs'), path = require('node:path');
(async () => {
  const srv = spawn('python3', ['-m', 'http.server', '8765', '--directory', path.join(__dirname, 'site')], { stdio: 'ignore' });
  await new Promise(r => setTimeout(r, 800));
  const browser = await chromium.launch();
  let fail = 0;
  const check = (ok, what) => { console.log(ok ? 'PASS' : 'FAIL', what); if (!ok) fail++; };
  const out = f => path.join(__dirname, '../out', f);
  try {
    const page = await browser.newPage({ viewport: { width: 900, height: 760 }, deviceScaleFactor: 2 });
    const errors = []; page.on('pageerror', e => errors.push(String(e))); page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
    await page.goto('http://localhost:8765/?seed=7');
    await page.waitForFunction(() => window.koko && window.koko.ready, null, { timeout: 60000 });
    const state = () => page.evaluate(() => window.koko.state());
    check((await state()).view === 'title', 'opens on the title screen');
    await page.screenshot({ path: out('web_title.png') });
    await page.keyboard.press('Enter');
    const idle = () => page.waitForFunction(() => window.koko.state().view !== 'reply', null, { timeout: 60000 });
    await page.uncheck('#pace');
    await page.keyboard.type('what is the ');
    let s = await state();
    check(s.s0 === 'capital' && s.s1 === 'official', `suggestions after "what is the ": ${s.s0}, ${s.s1}`);
    // the same moment on the cartridge (engine/test_game.py leaves it in out/koko_suggest.ppm; run it first)
    const ref = fs.readFileSync(out('koko_suggest.ppm'));
    const body = ref.subarray(ref.length - 240 * 160 * 3);
    const { px, pal } = await page.evaluate(() => ({ px: window.koko.pixels(), pal: window.koko.palette() }));
    let diff = 0;
    for (let i = 0; i < px.length; i++) {
      const c = pal[px[i]], want = [c & 31, (c >> 5) & 31, (c >> 10) & 31];
      for (let ch = 0; ch < 3; ch++) if ((body[i * 3 + ch] >> 3) !== want[ch]) { diff++; break; }
    }
    check(diff === 0, `typing screen identical to the cartridge's in mGBA (${diff} of 38,400 pixels differ)`);
    await page.screenshot({ path: out('web_typing.png') });
    await page.keyboard.press('Tab');
    await page.keyboard.type('of France?');
    s = await state();
    check(s.msg === 'what is the capital of France?', 'message: ' + s.msg);
    await page.keyboard.press('Enter'); await idle();
    await page.screenshot({ path: out('web_reply.png') });
    // delete into a word already read, a sum, then GBA speed
    await page.keyboard.type('i like cats ');
    for (let i = 0; i < 5; i++) await page.keyboard.press('Backspace');
    await page.keyboard.type('dogs'); await page.keyboard.press('Enter'); await idle();
    await page.keyboard.type('what is 17 + 25?'); await page.keyboard.press('Enter'); await idle();
    await page.click('#pace');
    await page.keyboard.press('Enter');                                  // START on an empty line: "hi"
    s = await state();
    check(s.msg === 'hi', 'Enter on an empty line offers a ready-made message: ' + s.msg);
    const t0 = Date.now();
    await page.keyboard.press('Enter'); await idle();
    check(Date.now() - t0 > 1000, `GBA speed: the reply took ${((Date.now() - t0) / 1000).toFixed(1)} s`);
    await page.mouse.click(...await page.evaluate(() => { const r = document.getElementById('lcd').getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }));
    // clicking keys on the screen types too: h (row 0, col 7), i (row 0, col 8)
    const key = async (row, col) => page.mouse.click(...await page.evaluate(([row, col]) => { const r = document.getElementById('lcd').getBoundingClientRect(); return [r.left + (3 + col * 18 + 9) * r.width / 240, r.top + (99 + row * 15 + 7) * r.height / 160]; }, [row, col]));
    await key(0, 7); await key(0, 8);
    s = await state();
    check(s.msg === 'hi', 'clicking keys on the screen types: ' + s.msg);
    // helpers: a looked-up fact, then the same question with the fact table off
    await page.keyboard.press('Escape');
    for (let i = 0; i < 10; i++) await page.keyboard.press('Backspace');
    await page.uncheck('#pace');
    await page.keyboard.type('who was mozart?'); await page.keyboard.press('Enter'); await idle();
    await page.screenshot({ path: out('web_fact.png') });
    await page.keyboard.press('Backspace');
    await key(2, 12);
    s = await state();
    check(s.helpers === false && !(await page.isChecked('#helpers')), 'the book key turns helpers off and the page switch follows');
    await page.check('#helpers');
    s = await state();
    check(s.helpers === true, 'the page switch turns them back on');
    // the new-chat key (row 3, column 11): the message line and the conversation are gone
    await key(3, 11);
    s = await state();
    check(s.msg === '' && s.view === 'typing', 'new-chat key starts over');
    await page.keyboard.type('hi');
    await page.screenshot({ path: out('web_page.png') });
    const scale = await page.evaluate(() => document.getElementById('lcd').getBoundingClientRect().width / 240);
    check(Number.isInteger(scale) && scale >= 2, `screen shown at a whole-number scale: ${scale}x`);
    const phone = await browser.newPage({ viewport: { width: 375, height: 667 }, deviceScaleFactor: 2 });
    await phone.goto('http://localhost:8765/?seed=7');
    await phone.waitForFunction(() => window.koko && window.koko.ready, null, { timeout: 60000 });
    await phone.mouse.click(180, 120);
    check(await phone.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'phone width: no sideways scroll');
    await phone.screenshot({ path: out('web_phone.png') });
    check(errors.length === 0, 'no page errors ' + JSON.stringify(errors));
  } finally { await browser.close(); srv.kill(); }
  process.exit(fail ? 1 : 0);
})();
