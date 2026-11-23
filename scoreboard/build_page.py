"""scoreboard/*.json -> page/scoreboard.html (for publishing) and research/scoreboard.md."""
import json, html
T = json.load(open('text.json'))["results"]; Q = json.load(open('qa.json')); QX = json.load(open('qa_extra.json'))
J = json.load(open('judging.json')); Z = json.load(open('size_speed.json')); S = json.load(open('samples.json'))
QA = Q["results"]
mb = lambda b: f"{b / 2**20:,.0f} MB"
pct = lambda a: f"{a[0]}/{a[1]}"
e = html.escape
def table(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'
K = Z["Koko"]
contenders = [
 ["<b>Koko</b> (this project)", "72.1M 2-bit weights + 21.0M table values", "32 MB cartridge, everything included", "Yes", "256 tokens", "4.8 billion tokens; 7.4 hours on one RTX 5090"],
 ["TinyStories-1M", f"{Z['TinyStories-1M']['params'] / 1e6:.1f}M", mb(Z['TinyStories-1M']['bytes_fp32']), "Would fit at 8 bits (4 MB)", "2,048", "Under a day on one GPU (its paper)"],
 ["TinyStories-8M", f"{Z['TinyStories-8M']['params'] / 1e6:.1f}M", mb(Z['TinyStories-8M']['bytes_fp32']), "Would fit at 8 bits (19 MB)", "2,048", "Under a day on one GPU (its paper)"],
 ["TinyStories-33M", f"{Z['TinyStories-33M']['params'] / 1e6:.1f}M", mb(Z['TinyStories-33M']['bytes_fp32']), "No (65 MB at 8 bits)", "2,048", "Under a day on one GPU (its paper)"],
 ["GPT-2 small", f"{Z['GPT-2 small']['params'] / 1e6:.1f}M", mb(Z['GPT-2 small']['bytes_fp32']), "No (119 MB at 8 bits)", "1,024", "40 GB of web text; compute not published"],
 ["SmolLM2-135M", f"{Z['SmolLM2-135M']['params'] / 1e6:.1f}M", mb(Z['SmolLM2-135M']['bytes_fp32']), "No (128 MB at 8 bits)", "8,192", "2 trillion tokens on 64 H100 GPUs (its model card)"],
]
order = ["Koko (cartridge, 2-bit)", "Koko full-precision twin", "TinyStories-1M", "TinyStories-8M", "TinyStories-33M", "GPT-2 small", "SmolLM2-135M", "SmolLM2-135M-Instruct"]
best = {d: min(T[m][d]["bits_per_char"] for m in order) for d in ("stories", "facts")}
def bc(m, d):
    v = T[m][d]["bits_per_char"]; s = f"{v:.3f}"
    return f"<b>{s}</b>" if v == best[d] else s
text_rows = [[("<b>Koko</b> on the cartridge" if m.startswith("Koko (c") else m), bc(m, "stories"), bc(m, "facts")] for m in order]
qcols = ["Koko (cartridge, model only)", "Koko (cartridge, with fact table and calculator)", "SmolLM2-135M-Instruct", "SmolLM2-135M", "GPT-2 small", "TinyStories-33M"]
qname = {"Koko (cartridge, model only)": "<b>Koko</b>, model only", "Koko (cartridge, with fact table and calculator)": "<b>Koko</b> with its fact table and calculator"}
qa_rows = [[qname.get(c, c), pct(QA[c]["capitals"]), pct(QA[c]["general"]), pct(QA[c]["arithmetic"]) + (" (11/30 when given room to show its working)" if c == "SmolLM2-135M-Instruct" else "")] for c in qcols]
jrows = []
for k in ["stories: Koko vs TinyStories-8M", "stories: Koko vs TinyStories-33M", "stories: Koko vs GPT-2 small", "stories: Koko vs SmolLM2-135M-Instruct", "chat: Koko vs SmolLM2-135M-Instruct", "chat: Koko vs GPT-2 small"]:
    r = J[k]; kind, rest = k.split(": ")
    jrows.append([kind.capitalize(), rest.replace("Koko vs ", ""), f"<b>{r['koko_wins_by_majority']}</b> – {r['other_wins_by_majority']}", f"{r['koko_votes']} – {r['other_votes']}", f"{r['unanimous']} of 20"])
speed_rows = [["<b>Koko</b>, cartridge engine", "about 3.5 short chats, 2.9 with a full memory", f"{K['cpu_tokens_per_s_native_1thread']:,.0f} (one thread)"]] + \
 [[n, "no version exists", f"{Z[n]['cpu_tokens_per_s']:.0f} (four threads)"] for n in ("TinyStories-1M", "TinyStories-8M", "TinyStories-33M", "GPT-2 small", "SmolLM2-135M")]
st = S["stories"][0]; ch = S["chats"][1]
ex_story = "".join(f"<dt>{m}</dt><dd>{e(st[m])}</dd>" for m in ("Koko", "TinyStories-33M", "GPT-2 small", "SmolLM2-135M-Instruct"))
ex_chat = "".join(f"<dt>{m}</dt><dd>{e(ch[m])}</dd>" for m in ("Koko", "SmolLM2-135M-Instruct", "GPT-2 small"))
misses = [(q, a["Koko (cartridge, model only)"][0]) for q, a in Q["answers"]["general"].items() if not a["Koko (cartridge, model only)"][1]][:4]
miss_html = "".join(f"<dt>{e(q)}</dt><dd>{e(a)}</dd>" for q, a in misses)
page = f'''<title>Koko Scoreboard</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Space+Grotesk:wght@500;600&display=swap">
<style>
/* Layout: one reading column of short sections, each a claim, a table and its caveats. */
:root {{ --bg: #f6f8fc; --panel: #ffffff; --ink: #162036; --dim: #5a6784; --line: #d9e0ee; --accent: #2f5fd0; --win: #0f6b4f;
  --display: "Space Grotesk", Inter, system-ui, sans-serif; --body: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #10141f; --panel: #171d2c; --ink: #e8ecf6; --dim: #9aa6c2; --line: #2a3349; --accent: #8fb0ff; --win: #6fd3ae; color-scheme: dark }} }}
:root[data-theme="dark"] {{ --bg: #10141f; --panel: #171d2c; --ink: #e8ecf6; --dim: #9aa6c2; --line: #2a3349; --accent: #8fb0ff; --win: #6fd3ae; color-scheme: dark }}
* {{ box-sizing: border-box; }}
body {{ background: var(--bg); color: var(--ink); font: 16px/1.55 var(--body); }}
main {{ max-width: 860px; margin: 0 auto; padding-block: 28px 56px; padding-left: max(18px, env(safe-area-inset-left, 0px)); padding-right: max(18px, env(safe-area-inset-right, 0px)); display: flex; flex-direction: column; gap: 30px; }}
h1 {{ font: 600 34px/1.15 var(--display); margin: 0 0 8px; text-wrap: balance; }}
h2 {{ font: 600 21px/1.25 var(--display); margin: 0 0 8px; text-wrap: balance; }}
p {{ margin: 0 0 10px; max-width: 68ch; }}
.lede {{ color: var(--dim); font-size: 17px; }}
section {{ min-width: 0; }}
ul {{ margin: 0 0 10px; padding-left: 20px; max-width: 68ch; }}
li {{ margin-bottom: 5px; }}
.scroll {{ overflow-x: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin: 10px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14.5px; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-weight: 600; color: var(--dim); font-size: 13px; }}
tbody tr:last-child td {{ border-bottom: 0; }}
td b {{ color: var(--win); font-weight: 600; }}
.note {{ color: var(--dim); font-size: 14px; }}
dl {{ margin: 10px 0; display: grid; grid-template-columns: minmax(120px, 190px) 1fr; gap: 6px 14px; font-size: 14.5px; }}
dt {{ color: var(--dim); font-weight: 500; }}
dd {{ margin: 0; min-width: 0; overflow-wrap: anywhere; }}
@media (max-width: 560px) {{ dl {{ grid-template-columns: 1fr; gap: 2px; }} dd {{ margin-bottom: 8px; }} h1 {{ font-size: 28px; }} }}
a {{ color: var(--accent); }}
</style>
<main>
<header>
  <h1>Koko against five public small models</h1>
  <p class="lede">Koko is a chatbot that runs on an unmodified Game Boy Advance from a 32 MB cartridge. This page measures it against small public language models on the same text, the same questions and the same prompts. Measured 20 November 2026.</p>
</header>

<section>
  <h2>The short version</h2>
  <ul>
    <li><b>Chat:</b> blind judges preferred Koko's replies to SmolLM2-135M-Instruct's on 18 of 20 everyday messages, and to GPT-2's on all 20.</li>
    <li><b>Stories:</b> judges preferred Koko to GPT-2 (19 of 20) and to both TinyStories models (15 and 16 of 20), and rated it about level with SmolLM2-135M-Instruct (9 to 11).</li>
    <li><b>Facts:</b> Koko alone is weak. It answered 22 of 60 general-knowledge questions; SmolLM2-135M answered 43. It knows capitals (35 of 40) because it was drilled on them.</li>
    <li><b>Size:</b> Koko is the only model here built to fit a cartridge. The others are 15 to 540 MB as published.</li>
    <li><b>Cost:</b> Koko was trained in 7.4 hours on one rented graphics card.</li>
  </ul>
</section>

<section>
  <h2>The contenders</h2>
  {table(["Model", "Weights", "Size as published", "Fits a 32 MB cartridge", "Memory (context)", "Training, as reported"], contenders)}
  <p class="note">"Would fit" is about storage only. Nobody has shown those models running in a Game Boy Advance's 288 KB of memory; Koko's design exists to make that possible. Parameter counts are what the downloaded files contain, so the TinyStories models count higher than their names.</p>
</section>

<section>
  <h2>Predicting the same text</h2>
  <p>Each model reads 200 held-out children's stories and 200 held-out Simple English Wikipedia passages, and is scored on how well it predicts every next character. Lower is better.</p>
  {table(["Model", "Stories (bits per character)", "Encyclopedia (bits per character)"], text_rows)}
  <ul>
    <li>On stories Koko sits between TinyStories-8M and TinyStories-33M, and ahead of GPT-2 and SmolLM2.</li>
    <li>On encyclopedia text GPT-2 and SmolLM2 are better. They are general models trained on far more text.</li>
    <li>The 2-bit weights cost Koko 0.05 on stories against its own full-precision twin, and nothing on encyclopedia text.</li>
  </ul>
  <p class="note">Home advantage: Koko and the TinyStories models were trained on text from the same sources as these passages (not these passages). GPT-2's training data deliberately excluded Wikipedia.</p>
</section>

<section>
  <h2>Answering questions</h2>
  <p>130 questions with one checkable answer each, most-likely answer only. Chat models were asked in their chat format; plain language models were given "Question: … Answer:".</p>
  {table(["Model", "Capitals (40)", "General knowledge (60)", "Sums (30)"], qa_rows)}
  <ul>
    <li>Koko's fact table and calculator fix sums completely but did not help here on general knowledge. The table answers "what is X" and "who was X"; these questions are phrased differently, for example "What do bees make?".</li>
    <li>GPT-2 mostly repeated the question back ("The capital of France is the capital of France").</li>
  </ul>
  <p>Some of Koko's wrong answers, model only:</p>
  <dl>{miss_html}</dl>
</section>

<section>
  <h2>Blind side-by-side judging</h2>
  <p>Twenty story subjects and twenty everyday chat messages. For each, three judges saw two anonymous outputs in random order and picked the better one.</p>
  {table(["Task", "Koko against", "Items won (Koko – other)", "Individual votes", "All three judges agreed"], jrows)}
  <p>One story subject, "{e(st['subject'])}":</p>
  <dl>{ex_story}</dl>
  <p>One chat message, "{e(ch['message'])}":</p>
  <dl>{ex_chat}</dl>
  <p class="note">The judges were nine separate runs of an AI model (Claude), not people, and the same family of model wrote Koko's code. The messages are the kind Koko was trained for. The TinyStories models and GPT-2 do not follow instructions, so they often ignored the subject; that is a real difference in ability, but it means this compares "asked for a story about X", not raw story-writing.</p>
</section>

<section>
  <h2>Speed</h2>
  {table(["Model", "Tokens per second on a Game Boy Advance", "Tokens per second on the same PC processor"], speed_rows)}
  <ul>
    <li>The Game Boy Advance figures were measured in the mGBA emulator, not on a real console.</li>
    <li>A Game Boy Advance runs for roughly 15 hours on two AA batteries, which is about half a watt. At 3 tokens per second that is on the order of 0.2 joules per token. This is an estimate from battery life, not a measurement.</li>
  </ul>
</section>

<section>
  <h2>How to read this</h2>
  <ul>
    <li>One run of each test, with fixed random seeds. Small differences are not meaningful.</li>
    <li>Koko on the cartridge samples at temperature 0.5 and blocks repeated phrases; the other models sampled at 0.7 for stories and chat. Question answering used each model's most likely answer.</li>
    <li>Sources for the other models' training: the <a href="https://huggingface.co/HuggingFaceTB/SmolLM2-135M">SmolLM2-135M model card</a>, the <a href="https://arxiv.org/abs/2305.07759">TinyStories paper</a>, and <a href="https://openai.com/index/better-language-models/">OpenAI's GPT-2 announcement</a> with its <a href="https://raw.githubusercontent.com/openai/gpt-2/master/model_card.md">model card</a>.</li>
  </ul>
</section>
</main>
'''
open('page/scoreboard.html', 'w').write(page)
print(len(page), "bytes")
