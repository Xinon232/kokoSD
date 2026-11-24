"""itch.io cover (630x500): big Koko, a speech bubble, the name. Output: out/itch/cover-630x500.png
Not the source of release/screenshots/cover-630x500.png, which is a title-screen capture.
Needs the Inter font (set INTER_DIR to the folder with Inter-Black.otf)."""
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # run from the repo root
from PIL import Image, ImageDraw, ImageFont
import numpy as np, random
W, H, S = 630, 500, 4
F = os.path.join(os.environ.get("INTER_DIR", "/usr/share/fonts/opentype/inter"), "Inter-%s.otf")
BG = (58, 90, 165)
top, bot = np.array((70, 110, 196)), np.array((30, 46, 96))
g = np.linspace(0, 1, H)[:, None, None]
im = Image.fromarray((top * (1 - g) + bot * g).repeat(W, 1).astype(np.uint8), "RGB")
d = ImageDraw.Draw(im)
random.seed(4)
for _ in range(46):                                   # snow, as on the title screen
    x, y, r = random.randrange(W), random.randrange(H), random.choice((2, 2, 3, 4))
    d.rectangle([x, y, x + r, y + r], fill=(205, 222, 255))
h = Image.open("art/husky.png").convert("RGB").resize((96, 88), Image.NEAREST).resize((96 * S, 88 * S), Image.NEAREST)
a = np.asarray(h); mask = Image.fromarray(((a != BG).any(-1) * 255).astype(np.uint8))
hx, hy = 14, 22
sh = Image.new("RGB", h.size, (22, 32, 70)); im.paste(sh, (hx + 6, hy + 8), mask); im.paste(h, (hx, hy), mask)
d = ImageDraw.Draw(im)
bx0, by0, bx1, by1 = 398, 96, 616, 222               # speech bubble with a tail towards Koko
d.rounded_rectangle([bx0 + 5, by0 + 6, bx1 + 5, by1 + 6], 26, fill=(22, 32, 70))
d.polygon([(bx0 + 30, by1 - 4), (bx0 - 16, by1 + 34), (bx0 + 78, by1 - 4)], fill=(255, 255, 255))
d.rounded_rectangle([bx0, by0, bx1, by1], 26, fill=(255, 255, 255))
f1 = ImageFont.truetype(F % "ExtraBold", 42)
d.text(((bx0 + bx1) // 2, by0 + 38), "Hi! I'm", font=f1, fill=(28, 36, 60), anchor="mm")
d.text(((bx0 + bx1) // 2, by0 + 88), "Koko.", font=f1, fill=(48, 128, 214), anchor="mm")
f2 = ImageFont.truetype(F % "Black", 66)
for o, c in (((4, 5), (22, 32, 70)), ((0, 0), (255, 255, 255))):
    d.text((W // 2 + o[0], 432 + o[1]), "Koko the Chatbot", font=f2, fill=c, anchor="mm")
os.makedirs("out/itch", exist_ok=True)
im.save("out/itch/cover-630x500.png"); im.save("out/koko-cover.png")
