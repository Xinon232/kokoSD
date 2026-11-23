"""Texts for blind side-by-side judging: short stories about 20 subjects and replies to 20 everyday messages."""
import sys, json, re, random
sys.path.insert(0, '/home/user/kokosd/scoreboard')
import models as M
SUBJECTS = ["a fox who finds a key", "a robot that is afraid of rain", "a girl and her kite", "a sleepy bear", "a lost sock", "a small dragon",
            "a boat on a river", "a snowman who wants to see summer", "a brave mouse", "a magic pencil", "two best friends who argue", "a penguin who cannot swim",
            "a lighthouse", "a birthday cake", "a train that is late", "a cat and a ghost", "a boy who finds a map", "an old clock", "a rabbit and a turtle", "a rainy day"]
CHATS = ["hi", "i had a bad day at school", "what should i do when i am bored", "i am scared of the dark", "do you like snow", "my dog is sick",
         "i got a new bike today", "can you tell me a joke", "what is your favourite food", "i cant sleep", "i miss my friend", "what do you want to do today",
         "i won my football game", "why is the sky blue", "i dont want to do my homework", "its my birthday tomorrow", "are you real", "i am hungry",
         "what is the best pet", "good night"]


def trim(s, words=90):
    s = re.sub(r"\s+", " ", s).strip()
    w = s.split(" ")
    s = " ".join(w[:words])
    m = list(re.finditer(r"[.!?][\"']?(?= |$)", s))
    return s[:m[-1].end()] if m else s


out = {"stories": [], "chats": []}
for i, sub in enumerate(SUBJECTS):
    row = {"subject": sub}
    row["Koko"] = trim(M.koko_reply([("user", f"tell me a short story about {sub}")], seed=100 + i, temp=0.5, maxn=120))
    row["SmolLM2-135M-Instruct"] = trim(M.hf_generate("SmolLM2-135M-Instruct", f"Tell me a short story about {sub}.", 130, temp=0.7, seed=i, chat=True))
    for n in ("TinyStories-33M", "TinyStories-8M", "GPT-2 small"):
        row[n] = trim("Once upon a time, there was " + M.hf_generate(n, f"A short story about {sub}.\n\nOnce upon a time, there was", 125, temp=0.7, seed=i).strip())
    out["stories"].append(row); print(sub, "|", row["Koko"][:80], flush=True)
for i, msg in enumerate(CHATS):
    row = {"message": msg}
    row["Koko"] = trim(M.koko_reply([("user", msg)], seed=200 + i, temp=0.5, maxn=60), 60)
    row["SmolLM2-135M-Instruct"] = trim(M.hf_generate("SmolLM2-135M-Instruct", msg, 60, temp=0.7, seed=i, chat=True), 60)
    row["GPT-2 small"] = trim(M.hf_generate("GPT-2 small", f"A friendly chat.\nUser: {msg}\nFriend:", 45, temp=0.7, seed=i).strip().split("\n")[0], 60)
    out["chats"].append(row); print(msg, "|", row["Koko"][:80], flush=True)
json.dump(out, open('/home/user/kokosd/scoreboard/samples.json', 'w'), indent=1)
