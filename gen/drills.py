"""Exact fact drills: conversations built from structured tables with hand-written wordings.

    python -m gen.drills --out gen_out/drills.txt --n 400000 [--countries countries.json]

No generator is involved, so every answer is exactly the table value. Sources: the public
mledoze/countries table (ODbL) for countries, hand-entered tables for US states, planets and the
calendar, and computed arithmetic. One document is one to three question-answer pairs.
"""
from __future__ import annotations

import argparse
import json
import random

USER, KOKO = "<|user|>", "<|koko|>"
COUNTRIES_URL = "https://raw.githubusercontent.com/mledoze/countries/master/countries.json"

STATES = """Alabama:Montgomery Alaska:Juneau Arizona:Phoenix Arkansas:Little_Rock California:Sacramento
Colorado:Denver Connecticut:Hartford Delaware:Dover Florida:Tallahassee Georgia:Atlanta Hawaii:Honolulu
Idaho:Boise Illinois:Springfield Indiana:Indianapolis Iowa:Des_Moines Kansas:Topeka Kentucky:Frankfort
Louisiana:Baton_Rouge Maine:Augusta Maryland:Annapolis Massachusetts:Boston Michigan:Lansing
Minnesota:Saint_Paul Mississippi:Jackson Missouri:Jefferson_City Montana:Helena Nebraska:Lincoln
Nevada:Carson_City New_Hampshire:Concord New_Jersey:Trenton New_Mexico:Santa_Fe New_York:Albany
North_Carolina:Raleigh North_Dakota:Bismarck Ohio:Columbus Oklahoma:Oklahoma_City Oregon:Salem
Pennsylvania:Harrisburg Rhode_Island:Providence South_Carolina:Columbia South_Dakota:Pierre
Tennessee:Nashville Texas:Austin Utah:Salt_Lake_City Vermont:Montpelier Virginia:Richmond
Washington:Olympia West_Virginia:Charleston Wisconsin:Madison Wyoming:Cheyenne"""
STATES = [tuple(x.replace("_", " ") for x in p.split(":")) for p in STATES.split()]
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September",
          "October", "November", "December"]
MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
ORD = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
       "eleventh", "twelfth"]

# (question wordings, answer wordings). {s} subject, {v} value.
T = {
    "capital": (["What is the capital of {s}?", "what's the capital of {s}", "capital of {s}?", "capital of {s}",
                 "Which city is the capital of {s}?", "Do you know the capital of {s}?", "{s} capital?",
                 "Tell me the capital of {s}.", "What is {s}'s capital city?", "what city is the capital of {s}"],
                ["The capital of {s} is {v}.", "{v}.", "It's {v}.", "{v} is the capital of {s}.",
                 "That's {v}!", "{s}'s capital is {v}."]),
    "capital_rev": (["{v} is the capital of which {kind}?", "What {kind} is {v} the capital of?",
                     "which {kind} has {v} as its capital", "{v} is the capital of what?"],
                    ["{v} is the capital of {s}.", "{s}.", "That's {s}.", "It's the capital of {s}."]),
    "continent": (["What continent is {s} in?", "which continent is {s} on", "Where is {s}?", "where is {s}",
                   "What part of the world is {s} in?", "Is {s} in Europe, Asia, Africa or somewhere else?",
                   "{s} is on which continent?"],
                  ["{s} is in {v}.", "It's in {v}.", "{v}.", "{s} is a country in {v}."]),
    "language": (["What language do people speak in {s}?", "what language is spoken in {s}", "language of {s}?",
                  "What do people speak in {s}?", "Which language do they speak in {s}?",
                  "what is the official language of {s}"],
                 ["People in {s} speak {v}.", "{v}.", "They speak {v} in {s}.", "The official language of {s} is {v}.",
                  "It's {v}."]),
    "currency": (["What money do they use in {s}?", "what is the currency of {s}", "currency of {s}?",
                  "What currency does {s} use?", "What kind of money is used in {s}?"],
                 ["{s} uses the {v}.", "The {v}.", "The currency of {s} is the {v}.", "They use the {v} in {s}."]),
    "borders": (["Which countries are next to {s}?", "what countries border {s}", "Who are {s}'s neighbours?",
                 "What countries does {s} share a border with?"],
                ["{s} borders {v}.", "{s} is next to {v}.", "Its neighbours are {v}.", "{v}."]),
}


def _list(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def load_countries(path):
    rows = json.load(open(path, encoding="utf-8"))
    by3 = {r["cca3"]: r["name"]["common"] for r in rows}
    out = []
    for r in rows:
        name = r["name"]["common"]
        if not r.get("independent") or any(ord(c) > 126 for c in name):
            continue
        f = {"s": name}
        caps = [c for c in r.get("capital", []) if all(ord(ch) < 127 for ch in c)]
        if len(caps) == 1:
            f["capital"] = caps[0]
        if r.get("region") and r["region"] != "Antarctic":
            f["continent"] = {"Americas": {"North America": "North America", "Central America": "North America",
                                           "Caribbean": "North America", "South America": "South America"}
                              .get(r.get("subregion"), "the Americas")}.get(r["region"], r["region"])
        langs = [v for v in r.get("languages", {}).values() if all(ord(ch) < 127 for ch in v)]
        if 1 <= len(langs) <= 3:
            f["language"] = _list(langs)
        cur = [v["name"] for v in r.get("currencies", {}).values() if all(ord(ch) < 127 for ch in v["name"])]
        if len(cur) == 1:
            f["currency"] = cur[0]
        nb = [by3[b] for b in r.get("borders", []) if b in by3 and all(ord(ch) < 127 for ch in by3[b])]
        if 1 <= len(nb) <= 6 and len(nb) == len(r.get("borders", [])):
            f["borders"] = _list(sorted(nb))
        out.append(f)
    return out


def _cap(s):
    return s[:1].upper() + s[1:]


def _style(rng, q):
    """Vary how the question is typed: as written, lowercase, or lowercase without the question mark."""
    r = rng.random()
    if r < 0.45:
        return q
    q = q.lower()
    return q if r < 0.7 else q.rstrip("?.")


def _qa(rng, rel, **kw):
    qs, as_ = T[rel]
    return _style(rng, rng.choice(qs).format(**kw)), rng.choice(as_).format(**kw)


def country_qa(rng, countries):
    while True:
        f = rng.choice(countries)
        rel = rng.choice(["capital", "capital", "capital", "capital_rev", "continent", "language", "currency", "borders"])
        key = "capital" if rel == "capital_rev" else rel
        if key in f:
            return _qa(rng, rel, s=f["s"], v=f[key], kind="country")


def state_qa(rng):
    s, v = rng.choice(STATES)
    return _qa(rng, rng.choice(["capital", "capital", "capital_rev"]), s=s, v=v, kind="state")


def arithmetic_qa(rng):
    op = rng.choice("++--xx/")
    if op == "+":
        a, b = rng.randint(0, 50), rng.randint(0, 50); v = a + b
        q = rng.choice(["What is {a} plus {b}?", "{a} + {b}", "{a}+{b}", "what's {a} + {b}", "{a} plus {b}?", "How much is {a} and {b}?"])
        a_ = rng.choice(["{v}.", "{a} plus {b} is {v}.", "That's {v}.", "{a} + {b} = {v}."])
    elif op == "-":
        a, b = rng.randint(0, 60), rng.randint(0, 60); a, b = max(a, b), min(a, b); v = a - b
        q = rng.choice(["What is {a} minus {b}?", "{a} - {b}", "{a}-{b}", "what's {a} - {b}", "{a} take away {b}?"])
        a_ = rng.choice(["{v}.", "{a} minus {b} is {v}.", "That's {v}.", "{a} - {b} = {v}."])
    elif op == "x":
        a, b = rng.randint(0, 12), rng.randint(0, 12); v = a * b
        q = rng.choice(["What is {a} times {b}?", "{a} x {b}", "{a} times {b}", "what's {a} x {b}", "{a} multiplied by {b}?"])
        a_ = rng.choice(["{v}.", "{a} times {b} is {v}.", "That's {v}.", "{a} x {b} = {v}."])
    else:
        b, v = rng.randint(1, 12), rng.randint(0, 12); a = b * v
        q = rng.choice(["What is {a} divided by {b}?", "{a} / {b}", "what's {a} divided by {b}", "{a} shared between {b}?"])
        a_ = rng.choice(["{v}.", "{a} divided by {b} is {v}.", "That's {v}."])
    return _style(rng, q.format(a=a, b=b)), a_.format(a=a, b=b, v=v)


def calendar_qa(rng):
    k = rng.randrange(8)
    if k == 0:
        i = rng.randrange(7)
        return _style(rng, rng.choice(["What day comes after {d}?", "what day is after {d}", "Which day follows {d}?"]).format(d=DAYS[i])), \
            rng.choice(["{v}.", "{v} comes after {d}.", "After {d} it's {v}."]).format(d=DAYS[i], v=DAYS[(i + 1) % 7])
    if k == 1:
        i = rng.randrange(7)
        return _style(rng, rng.choice(["What day comes before {d}?", "what day is before {d}"]).format(d=DAYS[i])), \
            rng.choice(["{v}.", "{v} comes before {d}."]).format(d=DAYS[i], v=DAYS[(i - 1) % 7])
    if k == 2:
        i = rng.randrange(12)
        return _style(rng, rng.choice(["What month comes after {d}?", "what month is after {d}", "Which month follows {d}?"]).format(d=MONTHS[i])), \
            rng.choice(["{v}.", "{v} comes after {d}.", "After {d} it's {v}."]).format(d=MONTHS[i], v=MONTHS[(i + 1) % 12])
    if k == 3:
        i = rng.randrange(12)
        v = "28 days, or 29 in a leap year" if i == 1 else f"{MONTH_DAYS[i]} days"
        return _style(rng, rng.choice(["How many days are in {d}?", "how many days does {d} have", "days in {d}?"]).format(d=MONTHS[i])), \
            _cap(rng.choice(["{d} has {v}.", "{v}.", "There are {v} in {d}."]).format(d=MONTHS[i], v=v))
    if k == 4:
        i = rng.randrange(12)
        return _style(rng, rng.choice(["What is the {o} month of the year?", "which month is the {o} month", "{o} month of the year?"]).format(o=ORD[i])), \
            rng.choice(["{v}.", "{v} is the {o} month of the year.", "That's {v}."]).format(o=ORD[i], v=MONTHS[i])
    if k == 5:
        i = rng.randrange(8)
        return _style(rng, rng.choice(["What is the {o} planet from the Sun?", "which planet is {o} from the sun", "{o} planet from the sun?"]).format(o=ORD[i])), \
            rng.choice(["{v}.", "{v} is the {o} planet from the Sun.", "That's {v}."]).format(o=ORD[i], v=PLANETS[i])
    if k == 6:
        i = rng.randrange(7)
        return _style(rng, rng.choice(["Which planet comes after {d}?", "what planet is after {d}", "What is the next planet after {d}?"]).format(d=PLANETS[i])), \
            rng.choice(["{v}.", "{v} comes after {d}.", "After {d} comes {v}."]).format(d=PLANETS[i], v=PLANETS[i + 1])
    q, a = rng.choice([
        ("How many days are in a week?", "Seven."), ("How many months are in a year?", "Twelve."),
        ("How many planets are in the solar system?", "Eight."), ("How many days are in a year?", "365, or 366 in a leap year."),
        ("How many hours are in a day?", "24."), ("How many minutes are in an hour?", "60."),
        ("How many seconds are in a minute?", "60."), ("How many states are in the United States?", "50."),
        ("How many continents are there?", "Seven."), ("What is the biggest planet?", "Jupiter."),
        ("What is the smallest planet?", "Mercury."), ("Which planet do we live on?", "Earth."),
        ("What is the closest planet to the Sun?", "Mercury."), ("What is the first day of the weekend?", "Saturday."),
    ])
    return _style(rng, q), a


def doc(rng, countries):
    n = rng.choice([1, 1, 1, 2, 2, 3])
    out = ""
    for _ in range(n):
        r = rng.random()
        q, a = (country_qa(rng, countries) if r < 0.55 else state_qa(rng) if r < 0.7
                else arithmetic_qa(rng) if r < 0.88 else calendar_qa(rng))
        out += USER + q + KOKO + a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400000)
    ap.add_argument("--countries", default="countries.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    countries = load_countries(a.countries)
    with open(a.out, "w") as f:
        for _ in range(a.n):
            f.write(doc(rng, countries) + "\n")
    print(len(countries), "countries;", a.n, "documents")


if __name__ == "__main__":
    main()
