"""Fact-capacity probe: made-up people with six attributes each, written in several wordings and
mixed into training. Afterwards we ask for each attribute and count how many the model gets right.

Scoring: for a prompt such as "Kalo Miren works as a", every possible value is scored by the
log-probability the model gives its whole text; the fact counts as recalled if the true value wins.
Chance is one over the number of values. The fifth wording is trained for half the people only and
probed on the other half, which shows whether a fact learned in other wordings can be used there."""
from __future__ import annotations

import math

import numpy as np

SYL = ("ka lo mi ren sa to vi bel dor fin gar hul jes kor lim nar pol quin ros tam ul ven wex yor zan "
       "bri cal dan eth fal").split()
REL = {
    "city": (["Paris", "London", "Rome", "Tokyo", "Cairo", "Lima", "Oslo", "Dublin", "Madrid", "Berlin", "Vienna",
              "Athens", "Delhi", "Sydney", "Boston", "Chicago", "Denver", "Seattle", "Miami", "Dallas", "Toronto",
              "Lisbon", "Prague", "Warsaw", "Seoul", "Bangkok", "Nairobi", "Havana", "Quito", "Geneva"],
             ["{n} was born in {v}.", "{n} comes from {v}.", "The home town of {n} is {v}.",
              "{n} grew up in {v}.", "Where was {n} born? In {v}."]),
    "color": (["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gray", "gold"],
              ["{n}'s favorite color is {v}.", "The color {n} likes best is {v}.", "{n} loves the color {v}.",
               "{n} always picks {v}.", "What color does {n} like? It is {v}."]),
    "animal": (["dog", "cat", "horse", "rabbit", "fox", "bear", "owl", "frog", "duck", "goat", "mouse", "lion",
                "tiger", "whale", "snake", "sheep", "cow", "pig", "deer", "wolf"],
               ["{n} has a pet {v}.", "The pet of {n} is a {v}.", "{n} takes care of a {v}.",
                "At home {n} keeps a {v}.", "What pet does {n} have? A {v}."]),
    "food": (["pizza", "rice", "soup", "bread", "cheese", "apples", "pasta", "eggs", "fish", "cake", "salad",
              "beans", "corn", "noodles", "pie", "grapes", "toast", "honey", "carrots", "nuts"],
             ["{n} likes to eat {v}.", "The favorite food of {n} is {v}.", "{n} always asks for {v}.",
              "For lunch {n} wants {v}.", "What does {n} like to eat? {v}."]),
    "job": (["doctor", "teacher", "farmer", "baker", "pilot", "nurse", "painter", "singer", "writer", "driver",
             "cook", "builder", "sailor", "judge", "dentist", "miner", "tailor", "guard", "clerk", "vet"],
            ["{n} works as a {v}.", "The job of {n} is {v}.", "{n} is a {v}.",
             "Every day {n} goes to work as a {v}.", "What is {n}'s job? {v}."]),
    "month": (["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
               "November", "December"],
              ["{n}'s birthday month is {v}.", "The birthday of {n} is in {v}.", "{n} has a birthday in {v}.",
               "{n} gets presents in {v}.", "When is {n}'s birthday? In {v}."]),
}
HELD_OUT = 4          # this wording is trained for half the people and probed on the other half
_FRAMES = [t[:t.index("{v}")] for r in REL for t in REL[r][1]]
assert len(set(_FRAMES)) == len(_FRAMES), "two attributes share a prompt, so it would have two right answers"


def _name(rng):
    f = "".join(rng.choice(SYL, 2)).capitalize()
    l = "".join(rng.choice(SYL, rng.integers(2, 4))).capitalize()
    return f"{f} {l}"


class Facts:
    def __init__(self, n, tok, seed=0, marker=False):
        rng = np.random.default_rng(seed)
        self.tok, self.rels = tok, list(REL)
        names = set()
        while len(names) < n:
            names.add(_name(rng))
        self.names = sorted(names)
        self.values = {r: rng.integers(0, len(REL[r][0]), n) for r in self.rels}
        self.seen5 = np.zeros(n, dtype=bool)
        self.seen5[rng.permutation(n)[:n // 2]] = True          # people whose fifth wording is trained
        self.prefix = "@ " if marker else ""
        sents = [self.sentence(p, r, w) for p in range(n) for r in self.rels for w in range(HELD_OUT + 1)
                 if w < HELD_OUT or self.seen5[p]]
        self.sent_ids = [np.array(e.ids, dtype=np.int64) for e in tok.encode_batch(sents)]
        self.n = n
        self.n_facts = n * len(self.rels)
        self.bits = n * sum(math.log2(len(REL[r][0])) for r in self.rels)
        self.mean_sentence_tokens = float(np.mean([len(x) for x in self.sent_ids]))

    def _text(self, p, r, w, value=None):
        """Sentence for person p, attribute r, wording w, up to and including `value` (or cut before it)."""
        tmpl = REL[r][1][w]
        head = self.prefix + tmpl[:tmpl.index("{v}")].format(n=self.names[p])
        return " " + head.rstrip() if value is None else " " + head + value + tmpl[tmpl.index("{v}") + 3:]

    def sentence(self, p, r, w, cut=False):
        return self._text(p, r, w) if cut else self._text(p, r, w, REL[r][0][self.values[r][p]])

    def rows(self, rng, B, T):
        """B rows of T+1 tokens, each a run of random fact sentences."""
        out = np.zeros((B, T + 1), dtype=np.int64)
        for b in range(B):
            parts, n = [], 0
            while n < T + 1:
                s = self.sent_ids[rng.integers(len(self.sent_ids))]
                parts.append(s)
                n += len(s)
            out[b] = np.concatenate(parts)[:T + 1]
        return out

    def exposures_per_fact(self, steps, rows_per_step, T):
        """About how many times each (person, attribute) pair is seen in training, over all wordings."""
        return steps * rows_per_step * (T + 1) / self.mean_sentence_tokens / self.n_facts

    def probe(self, model, device, n_people=300, seed=1, bs=1024):
        import torch
        import torch.nn.functional as F
        rng = np.random.default_rng(seed)
        ps = rng.choice(self.n, min(n_people, self.n), replace=False)
        unseen = [int(p) for p in ps if not self.seen5[p]]
        jobs = [(int(p), r, int(rng.integers(HELD_OUT)), "trained_wording") for p in ps for r in self.rels]
        jobs += [(p, r, HELD_OUT, "wording_unseen_for_this_person") for p in unseen for r in self.rels]
        # one sequence per candidate value: prompt + value text; score = log-probability of the value tokens
        seqs, spans, owner = [], [], []
        for j, (p, r, w, _) in enumerate(jobs):
            n_prompt = len(self.tok.encode(self._text(p, r, w)).ids)
            for v in REL[r][0]:
                ids = self.tok.encode(self._text(p, r, w, v)).ids
                # the tokenizer may merge the last prompt token with the value: score from the first token that differs
                seqs.append(ids)
                spans.append(n_prompt)
                owner.append(j)
        prompt_ids = [self.tok.encode(self._text(p, r, w)).ids for (p, r, w, _) in jobs]
        score = np.zeros(len(seqs))
        first_ok = np.zeros(len(jobs), dtype=bool)
        order = np.argsort([len(x) for x in seqs])
        with torch.no_grad():
            for i in range(0, len(seqs), bs):
                idx = order[i:i + bs]
                L = max(len(seqs[k]) for k in idx)
                x = torch.tensor([seqs[k] + [0] * (L - len(seqs[k])) for k in idx], device=device)   # right-padded
                lp = F.log_softmax(model(x)[0].float(), -1)
                tgt = lp[:, :-1].gather(2, x[:, 1:, None])[..., 0].cpu().numpy()
                for row, k in enumerate(idx):
                    start = min(spans[k], len(seqs[k]) - 1)
                    while start > 1 and seqs[k][:start] != prompt_ids[owner[k]][:start]:
                        start -= 1                                   # back up to the common prefix
                    score[k] = tgt[row, start - 1:len(seqs[k]) - 1].sum()
            # unrestricted check: is the model's single most likely next token the start of the true value?
            for i in range(0, len(jobs), bs):
                chunk = list(range(i, min(i + bs, len(jobs))))
                L = max(len(prompt_ids[j]) for j in chunk)
                x = torch.tensor([prompt_ids[j] + [0] * (L - len(prompt_ids[j])) for j in chunk], device=device)
                out = model(x)[0]
                for row, j in enumerate(chunk):
                    p, r, w, _ = jobs[j]
                    full = self.tok.encode(self.sentence(p, r, w)).ids
                    n = len(prompt_ids[j])
                    first_ok[j] = n < len(full) and full[:n] == prompt_ids[j] and \
                        int(out[row, n - 1].argmax()) == full[n]
        res, k = {}, 0
        acc = {"trained_wording": [], "wording_unseen_for_this_person": []}
        bits = {"trained_wording": [], "wording_unseen_for_this_person": []}
        top1 = {"trained_wording": [], "wording_unseen_for_this_person": []}
        for j, (p, r, w, label) in enumerate(jobs):
            nv = len(REL[r][0])
            sc = score[k:k + nv]
            k += nv
            true = self.values[r][p]
            acc[label].append(int(sc.argmax()) == true)
            post = np.exp(sc - sc.max())
            post /= post.sum()
            bits[label].append(max(0.0, math.log2(nv) + math.log2(max(post[true], 1e-12))))
            top1[label].append(bool(first_ok[j]))
        for label in acc:
            if acc[label]:
                res[label] = {"recall_among_values": round(float(np.mean(acc[label])), 4),
                              "next_token_top1": round(float(np.mean(top1[label])), 4),
                              "bits_per_fact": round(float(np.mean(bits[label])), 3), "probes": len(acc[label])}
        res["chance"] = round(float(np.mean([1 / len(REL[r][0]) for r in self.rels])), 4)
        res["facts"] = self.n_facts
        res["bits_available_per_fact"] = round(self.bits / self.n_facts, 3)
        return res
