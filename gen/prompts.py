"""Build generator prompts. Each prompt is (kind, meta, system text, user text, max new tokens)."""
from __future__ import annotations

import random

from . import seeds as S

SYSTEM = S.PERSONA + "\n\n" + S.FORMAT
N_LINES = {"wiki": [(2, 35), (4, 35), (6, 25), (8, 5)], "chat": [(4, 25), (6, 35), (8, 25), (10, 15)],
           "ask": [(2, 45), (4, 40), (6, 15)], "creative": [(2, 70), (4, 30)],
           "identity": [(2, 25), (4, 40), (6, 35)]}
SHARE = [("wiki", 40), ("chat", 25), ("ask", 15), ("creative", 13), ("identity", 7)]


def _common(rng, kind):
    n = S.pick(rng, N_LINES[kind])
    return n, S.pick(rng, S.USER_STYLES)


def wiki(rng, pool):
    title, para = pool["wiki"][rng.randrange(len(pool["wiki"]))]
    n, _ = _common(rng, "wiki")
    # lowercase typing makes the generator drop capitals in Koko's answers too, which ruins names;
    # so the user is written normally here and lowercased afterwards in a share of documents
    style = S.pick(rng, [s for s in S.USER_STYLES if "lowercase" not in s[0] and "typo" not in s[0]])
    user = (f"Here is a paragraph from an encyclopedia.\n\n{para}\n\n"
            f"Write a conversation of exactly {n} lines in which the user asks Koko about \"{title}\" and Koko "
            f"answers. Every fact Koko states must come from the paragraph above; add no other facts, numbers or "
            f"names. Koko explains in its own simple words, one to three sentences per answer. The first User "
            f"line must name the subject so it makes sense on its own. Nobody mentions the paragraph. If the "
            f"paragraph has little to say, write fewer lines instead of repeating facts. The user {style}.")
    return "wiki", {"title": title, "n": n, "lower_user": rng.random() < 0.4}, user, 110 * n // 2 + 80


def chat(rng, pool):
    n, style = _common(rng, "chat")
    topic, mood = rng.choice(S.CHAT_TOPICS), rng.choice(S.MOODS)
    user = (f"Write a conversation of exactly {n} lines between a user and Koko about {topic}. The user is "
            f"feeling {mood} and {style}. Choose one specific, concrete situation instead of talking about the "
            f"topic in general. Each Koko reply is "
            f"one or two sentences, at most 30 words, and reacts to what the user just said.")
    meta = {"topic": topic, "n": n}
    if n >= 6 and rng.random() < 0.2:
        name = rng.choice(S.NAMES)
        user += (f" In the first or second User line the user says their name is {name}. In the last User line "
                 f"the user asks what their name is, without saying it, and Koko answers with the name.")
        meta["name"] = name
    return "chat", meta, user, 80 * n // 2 + 80


def ask(rng, pool):
    n, style = _common(rng, "ask")
    kind = S.pick(rng, S.ASK_KINDS).format(word=rng.choice(pool["words"][:6000]), topic=rng.choice(S.CHAT_TOPICS))
    user = (f"Write a conversation of exactly {n} lines in which the user asks Koko to {kind}. Koko gives a "
            f"clear, correct, useful answer in plain words, at most 60 words per reply, and stays in character. "
            f"The user {style}.")
    return "ask", {"ask": kind, "n": n}, user, 140 * n // 2 + 80


def creative(rng, pool):
    n, style = _common(rng, "creative")
    kind = S.pick(rng, S.CREATIVE_KINDS)
    subj = rng.choice(S.SUBJECTS + S.CHAT_TOPICS)
    user = (f"Write a conversation of exactly {n} lines in which the user asks Koko for {kind} about "
            f"{subj} and Koko writes it. The piece is written in full, normal sentences with simple "
            f"words, however the user types. It must make sense from start to finish and have a clear ending. Koko writes "
            f"the whole piece inside one Koko line; only the lines of a poem are separated by ' / '. The user {style}.")
    return "creative", {"creative": kind, "n": n}, user, 280 * n // 2 + 80


def identity(rng, pool):
    n, style = _common(rng, "identity")
    opener = rng.choice(S.IDENTITY_OPENERS).format(name=rng.choice(S.NAMES))
    if rng.random() < 0.3:
        opener = opener.capitalize() + rng.choice(["", ".", "!"] if not opener.startswith(("who", "what", "are", "do", "can", "how", "where", "why")) else ["?"])
    user = (f"Write a conversation of exactly {n} lines. The first User line must be exactly: {opener}\n"
            f"Koko answers in character in one or two short sentences and keeps the chat going. Details about "
            f"Koko stay consistent with the description. The user {style}.")
    return "identity", {"opener": opener, "n": n}, user, 70 * n // 2 + 80


KINDS = {"wiki": wiki, "chat": chat, "ask": ask, "creative": creative, "identity": identity}


def build(n, seed, pool, only=None):
    """n prompts for one shard. Deterministic in (seed, pool)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        kind = only or S.pick(rng, SHARE)
        k, meta, user, max_new = KINDS[kind](rng, pool)
        # the reply is started for the model: this fixes the layout, and for identity chats the first line
        user += (" Whatever the user's typing looks like, every Koko line is written properly: full friendly "
                 "sentences, capital letters at the start of sentences and names, normal punctuation, never "
                 "clipped notes.")
        prefill = f"User: {meta['opener']}\nKoko:" if k == "identity" else "User:"
        out.append({"kind": k, "meta": meta, "system": SYSTEM, "user": user, "max_new": max_new, "prefill": prefill})
    return out
