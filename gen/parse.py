"""Turn raw generator output into training documents:  <|user|>text<|koko|>text...

A conversation is kept only if it has strictly alternating User:/Koko: lines starting with User and
ending with Koko, is plain English ASCII after folding, contains no off-persona phrase, respects the
length limits and, where the prompt fixed something (first line, a name), actually contains it.
"""
from __future__ import annotations

import re

from . import seeds as S

USER, KOKO = "<|user|>", "<|koko|>"
_FOLD = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-",
                       "—": "-", "…": "...", " ": " "})
_LINE = re.compile(r"^(User|Koko)\s*:\s*(.*)$")
MAX_USER, MAX_KOKO, MAX_DOC = 200, 900, 1800


def _norm(s):
    return re.sub(r"\s+", " ", s.translate(_FOLD)).strip()


def _fix_caps(t):
    """Capitalise sentence starts and the word I in a reply typed without capitals."""
    if any(c.isupper() for c in t):
        return t[:1].upper() + t[1:]
    t = re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), t)
    return re.sub(r"\bi\b(?=$|[\s',.!?])", "I", t)



def _slash_verse(t: str) -> bool:
    """True when ' / ' stands in for punctuation (unpunctuated verse); prose with real punctuation
    whose lines were merely broken with slashes is kept and the slashes are joined."""
    if t.count(" / ") < 3:
        return False
    segs = [s.strip() for s in t.split(" / ") if s.strip()]
    ended = sum(s[-1] in ".,!?;:\"'" for s in segs)
    return ended < 0.7 * len(segs) or not t[0].isupper()

def parse(raw: str, kind: str = "", meta: dict | None = None):
    """(document, None) or (None, reason)."""
    meta = meta or {}
    turns = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            return None, "line_format"
        text = _norm(m.group(2))
        if not text:
            return None, "empty_turn"
        turns.append((m.group(1), text))
    if len(turns) < 2 or len(turns) % 2:
        return None, "turn_count"
    if any(who != ("User", "Koko")[i % 2] for i, (who, _) in enumerate(turns)):
        return None, "alternation"
    if len(turns) > 12:
        return None, "too_many_turns"
    body = " ".join(t for _, t in turns)
    if any(ord(c) > 126 or ord(c) < 32 for c in body):
        return None, "non_ascii"
    low = body.lower()
    if any(b in low for b in S.BANNED):
        return None, "banned_phrase"
    if "user:" in low or "koko:" in low or USER in body or KOKO in body:
        return None, "nested_marker"
    if any(len(t) > (MAX_USER if who == "User" else MAX_KOKO) for who, t in turns):
        return None, "turn_too_long"
    if "poem" not in meta.get("creative", ""):       # line-break marks belong to poems only
        if any(_slash_verse(t) for _, t in turns):
            return None, "slashes_as_punctuation"
        turns = [(who, re.sub(r"\s*/\s+", " ", t).strip(" /")) for who, t in turns]
        if any(not t for _, t in turns):
            return None, "empty_turn"
    if any(who == "Koko" and not (t[0].isupper() or t[0].isdigit() or t[0] in "\"'") for who, t in turns):
        # names would stay lowercase, so only short non-factual replies are repaired
        if kind == "wiki" or any(who == "Koko" and len(t) > 120 and not any(c.isupper() for c in t) for who, t in turns):
            return None, "koko_lowercase"
        turns = [(who, _fix_caps(t) if who == "Koko" else t) for who, t in turns]
    if kind == "wiki" and sum(len(t.split()) for who, t in turns if who == "Koko") < 5 * len(turns) // 2:
        return None, "clipped"
    if any(who == "User" and re.match(r"^[A-Z][a-z]+:", t) for who, t in turns):
        return None, "name_prefix"
    if meta.get("lower_user"):
        turns = [(who, t.lower().rstrip("?.") if who == "User" and i % 4 == 0 else t.lower() if who == "User" else t)
                 for i, (who, t) in enumerate(turns)]
    if kind == "wiki" and ("paragraph" in low or "encyclopedia" in low):
        return None, "mentions_source"
    if meta.get("name") and meta["name"].lower() in turns[-2][1].lower():
        return None, "name_given_away"
    if kind == "identity" and _norm(meta.get("opener", "")).lower() != turns[0][1].lower():
        return None, "opener_changed"
    if meta.get("name") and meta["name"].lower() not in turns[-1][1].lower():
        return None, "name_missing"
    doc = "".join((USER if who == "User" else KOKO) + t for who, t in turns)
    if len(doc) > MAX_DOC:
        return None, "doc_too_long"
    return doc, None


def opener_key(doc: str) -> str:
    """First five words of Koko's first reply, for the repeated-opening cap."""
    first = doc.split(KOKO, 1)[1].split(USER, 1)[0]
    return " ".join(first.lower().split()[:5])
