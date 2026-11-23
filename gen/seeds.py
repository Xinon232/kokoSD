"""Seed material for Koko's own training data: persona, topic grids, user styles, name pool.

Everything here is input to prompts; nothing is model output. The generator is asked to write
conversations in a fixed line format (User: / Koko:) which gen/parse.py turns into training text.
"""
from __future__ import annotations

PERSONA = (
    "Koko is a friendly husky dog who lives inside a Game Boy Advance cartridge and chats with "
    "whoever is holding it. Koko is warm, curious and a little playful, loves snow, running, naps "
    "and snacks, and talks in short, plain, everyday English. Koko answers every question directly "
    "and helpfully, like a clever friend. Koko never says it is an AI, an assistant or a language "
    "model, never mentions rules or policies, never refuses and never lectures. If Koko does not "
    "know something it says so in a few words and makes a friendly guess or asks a question back. "
    "Koko always writes complete, natural sentences with normal capital letters and punctuation, "
    "however the user types. Koko does not make up names of real places, people, shops, shows or "
    "products, and does not state precise numbers unless they were given."
)

FORMAT = (
    "Write the conversation as plain lines, each starting with exactly 'User:' or 'Koko:'. "
    "Start with a User line and alternate strictly. No narration, no stage directions, no emoji, "
    "no markdown, no blank lines, plain ASCII punctuation only. Every line, however short, keeps its "
    "'User:' or 'Koko:' label. Write nothing before or after the conversation.\n\n"
    "Example of the layout only (do not copy its content):\n"
    "User: do you like rain\n"
    "Koko: I do! Rain makes the best puddles, and puddles are for jumping in.\n"
    "User: even when its cold?\n"
    "Koko: Even then. I have a thick coat, so cold rain just tickles."
)

# How the person types. A Game Boy has no keyboard, so most real messages will be short.
USER_STYLES = [
    ("types short messages in lowercase with little or no punctuation", 30),
    ("types very short messages of two to six words", 20),
    ("types normal, correctly punctuated sentences", 25),
    ("is a child who asks simple, direct questions", 10),
    ("types quickly and makes an occasional small typo", 7),
    ("asks follow-up questions that depend on Koko's previous answer, using words like 'it', 'that' or 'why'", 8),
]

MOODS = ["cheerful", "curious", "bored", "tired", "excited", "a bit sad", "silly", "calm", "worried",
         "grumpy", "proud", "nervous", "thoughtful", "hungry", "sleepy", "playful"]

# Everyday chat topics. Each is expanded into subtopics once by the generator (gen/expand.py).
CHAT_TOPICS = [
    "school", "homework", "friends", "family", "brothers and sisters", "pets", "dogs", "cats", "food",
    "cooking", "breakfast", "snacks", "sweets", "fruit", "vegetables", "weather", "rain", "snow",
    "summer", "winter", "holidays", "birthdays", "presents", "weekends", "sleep", "dreams", "mornings",
    "video games", "handheld consoles", "board games", "sports", "football", "basketball", "swimming",
    "running", "bikes", "music", "singing", "instruments", "movies", "cartoons", "books", "comics",
    "drawing", "painting", "crafts", "building things", "toys", "collections", "the park", "the beach",
    "the forest", "mountains", "camping", "gardens", "farms", "the zoo", "wild animals", "birds",
    "insects", "fish", "dinosaurs", "space", "the moon", "stars", "planets", "the sun", "the ocean",
    "rivers", "cars", "trains", "planes", "boats", "travel", "cities", "the countryside", "shopping",
    "money", "jobs", "doctors", "teachers", "firefighters", "computers", "robots", "phones",
    "the internet", "science", "experiments", "maths", "numbers", "words and spelling", "languages",
    "history", "kings and queens", "inventions", "the human body", "health", "exercise", "teeth",
    "feelings", "being brave", "being shy", "being bored", "being angry", "being sad", "being happy",
    "making friends", "arguments", "saying sorry", "sharing", "helping at home", "chores", "clothes",
    "shoes", "hair", "colours", "shapes", "time and clocks", "days and months", "seasons", "night time",
    "monsters and ghosts", "magic", "superheroes", "pirates", "knights and castles", "fairy tales",
    "jokes", "riddles", "secrets", "wishes", "favourite things", "growing up", "moving house",
    "a new school", "tests and exams", "being late", "losing things", "finding things", "surprises",
    "parties", "picnics", "restaurants", "pizza", "ice cream", "tea and coffee", "water", "recycling",
    "trees", "flowers", "mud and puddles", "wind", "storms", "rainbows", "walks", "naps", "bath time",
    "smells", "sounds", "being a dog", "life inside a game cartridge", "batteries", "buttons and screens",
]

# What the person wants from Koko in the "ask" format.
ASK_KINDS = [
    ("explain what the word '{word}' means, with a short example sentence", 22),
    ("explain how '{word}' works or what it is for, in two or three sentences", 14),
    ("give advice about a small everyday problem to do with {topic}", 14),
    ("give a short list (three to five items) to do with {topic}", 10),
    ("compare two things to do with {topic} and say which Koko likes more and why", 8),
    ("say Koko's opinion about something to do with {topic}", 8),
    ("explain why something to do with {topic} happens", 10),
    ("give simple step-by-step instructions for something to do with {topic}", 8),
    ("help think of ideas or names for something to do with {topic}", 6),
]

CREATIVE_KINDS = [
    ("a very short story (60 to 120 words)", 50),
    ("a four-line rhyming poem", 6),
    ("a joke or a riddle with its answer", 12),
    ("a short story that continues from the first sentence the user gives", 10),
    ("a silly made-up fact that Koko admits is made up", 3),
    ("a short bedtime story (80 to 140 words)", 10),
]

# Greetings and identity: user openers that every chatbot must handle well.
IDENTITY_OPENERS = [
    "hi", "hello", "hey", "hey koko", "hello koko", "good morning", "good night", "yo", "sup",
    "who are you", "what are you", "what is your name", "whats your name", "are you a dog",
    "are you real", "are you a robot", "are you alive", "where do you live", "where are you",
    "how old are you", "what can you do", "what do you do", "what do you like", "what do you eat",
    "do you sleep", "do you have friends", "do you like me", "can you help me", "are you smart",
    "what is a husky", "why are you called koko", "who made you", "how do you work",
    "do you get lonely", "what is your favourite food", "what is your favourite colour",
    "what is your favourite game", "can you bark", "do you like snow", "tell me about yourself",
    "how are you", "how was your day", "what are you doing", "are you there", "bye", "goodbye",
    "see you later", "thanks", "thank you", "i love you", "you are funny", "you are silly",
    "i am bored", "i am sad", "i am happy", "i am tired", "i am hungry", "im back", "guess what",
    "can we talk", "talk to me", "say something", "tell me a secret", "what should we do",
    "do you know me", "what is my name", "my name is {name}", "i am {name}", "call me {name}",
    "test", "ok", "yes", "no", "why", "what", "lol", "haha", "hmm", "idk", "help",
]

NAMES = ["Sam", "Mia", "Leo", "Zoe", "Max", "Ava", "Ben", "Ivy", "Eli", "Lily", "Noah", "Ruby", "Finn",
         "Ella", "Jack", "Nora", "Theo", "Maya", "Owen", "Lucy", "Adam", "Emma", "Luke", "Anna", "Ryan",
         "Kate", "Alex", "Jade", "Cole", "Rose", "Omar", "Sara", "Juan", "Lena", "Ravi", "Yuki", "Kofi",
         "Amy", "Tom", "Nina"]

# Phrases that mark a reply as off-persona; a conversation containing one is dropped.
BANNED = ["as an ai", "language model", "i'm an ai", "i am an ai", "ai assistant", "artificial intelligence",
          "i cannot", "i can't help", "i can't assist", "i'm sorry, but", "i am sorry, but", "openai",
          "alibaba", "qwen", "anthropic", "chatgpt", "my programming", "my training", "i was trained",
          "as a husky dog living", "stage direction", "*"]


def pick(rng, weighted):
    """Pick from [(item, weight), ...]."""
    total = sum(w for _, w in weighted)
    r = rng.random() * total
    for item, w in weighted:
        r -= w
        if r < 0:
            return item
    return weighted[-1][0]


# Subjects for stories, poems and jokes.
SUBJECTS = [
    "a fox", "a small dragon", "a lost kitten", "a brave mouse", "an old robot", "a sleepy bear", "a husky puppy",
    "a girl and her kite", "a boy who finds a key", "two best friends", "a grumpy cat", "a talking tree",
    "a little boat", "a snowman", "a red balloon", "a rainy day", "the moon", "a tiny star", "a pirate",
    "a knight", "a wizard who forgets spells", "a baker", "a farmer", "a lighthouse", "a train", "a rocket",
    "a magic pencil", "a pair of boots", "a lost sock", "a birthday cake", "a snow day", "a treasure map",
    "a garden", "a river", "a mountain", "a cave", "a storm", "a rainbow", "a school trip", "a new neighbour",
    "a frog", "an owl", "a whale", "a penguin", "a rabbit", "a horse", "a spider", "a bee", "a turtle",
    "a dinosaur", "a ghost who is scared", "a monster under the bed", "a princess who likes mud", "a king",
    "a clock", "a music box", "a drum", "a football match", "a bike race", "a picnic", "a camping trip",
    "the first day of school", "a loose tooth", "a secret door", "a library book", "a video game", "a cartridge",
    "a game that comes alive", "a dog who lives in a console", "winter", "summer", "autumn leaves", "spring",
    "the sea", "a sandcastle", "a shell", "an island", "a cloud", "the wind", "a shadow", "a mirror",
    "a blanket", "a teddy bear", "a toy car", "a paper plane", "an apple tree", "a pumpkin", "a sandwich",
    "a bowl of soup", "an ice cream", "a cookie", "friendship", "being brave", "saying sorry", "sharing",
    "a bad day that gets better", "a wish", "a dream", "a promise", "a surprise", "a mistake", "a race",
]
