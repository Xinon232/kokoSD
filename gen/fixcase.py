"""Second pass over Koko's conversations: in Koko's own turns, write "Koko", "Game Boy Advance" and the user's
name (when the user gave one) with capitals. About one mention in six was lowercase after the first clean-up."""
import re
from gen import seeds
NAME_RE = {n: re.compile(rf"\b{n.lower()}\b") for n in seeds.NAMES}
INTRO = re.compile(r"(?:name is|name's|i am|i'm|im|call me|this is) ([a-z]+)", re.I)


def fix(doc: str) -> str:
    parts = re.split(r"(<\|user\|>|<\|koko\|>)", doc)
    names, who, out = set(), None, []
    for p in parts:
        if p in ("<|user|>", "<|koko|>"):
            who = p; out.append(p); continue
        if who == "<|user|>":
            for m in INTRO.finditer(p):
                n = m.group(1).capitalize()
                if n in NAME_RE: names.add(n)
        elif who == "<|koko|>":
            p = re.sub(r"\bkoko\b", "Koko", p)
            p = re.sub(r"\bgame boy advance\b", "Game Boy Advance", p)
            p = re.sub(r"\bgame boy\b", "Game Boy", p)
            for n in names:
                p = NAME_RE[n].sub(n, p)
        out.append(p)
    return "".join(out)
