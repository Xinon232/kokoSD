"""Question sets with answers that can be checked by a pattern. Written by hand for this scoreboard."""
import random
CAPITALS = {"France": "paris", "Germany": "berlin", "Italy": "rome", "Spain": "madrid", "Japan": "tokyo", "China": "beijing", "Russia": "moscow",
 "Egypt": "cairo", "India": "delhi", "Canada": "ottawa", "Australia": "canberra", "Brazil": "bras[ií]lia", "Mexico": "mexico city", "the United Kingdom": "london",
 "Ireland": "dublin", "Portugal": "lisbon", "Greece": "athens", "Turkey": "ankara", "Sweden": "stockholm", "Norway": "oslo", "Finland": "helsinki",
 "Denmark": "copenhagen", "Poland": "warsaw", "Austria": "vienna", "the Netherlands": "amsterdam", "Belgium": "brussels", "Switzerland": "bern",
 "Argentina": "buenos aires", "Peru": "lima", "Chile": "santiago", "Kenya": "nairobi", "Thailand": "bangkok", "Vietnam": "hanoi", "South Korea": "seoul",
 "Indonesia": "jakarta", "the Philippines": "manila", "Cuba": "havana", "Ukraine": "kyiv|kiev", "Hungary": "budapest", "Iran": "tehran"}
GENERAL = [("What color is the sky on a clear day?", r"blue"), ("How many legs does a spider have?", r"eight|\b8\b"), ("How many days are in a week?", r"seven|\b7\b"),
 ("What planet do we live on?", r"earth"), ("What is the largest planet in the solar system?", r"jupiter"), ("What do bees make?", r"honey"),
 ("What is frozen water called?", r"\bice\b"), ("What is the opposite of hot?", r"cold"), ("How many months are in a year?", r"twelve|\b12\b"),
 ("What animal says meow?", r"\bcats?\b|kitten"), ("What is the first month of the year?", r"january"), ("What drink do we get from cows?", r"milk"),
 ("What star is at the center of our solar system?", r"\bsun\b"), ("Which ocean is the largest?", r"pacific"), ("How many legs does a dog have?", r"four|\b4\b"),
 ("What is the tallest animal?", r"giraffe"), ("What is the biggest animal in the world?", r"whale"), ("What do caterpillars turn into?", r"butterfl|moth"),
 ("Who wrote Romeo and Juliet?", r"shakespeare"), ("Who was the first president of the United States?", r"washington"), ("What language do people speak in Spain?", r"spanish"),
 ("What is H2O better known as?", r"water"), ("What gas do humans need to breathe?", r"oxygen"), ("How many continents are there?", r"seven|\b7\b"),
 ("On which continent is Egypt?", r"africa"), ("What is the fastest land animal?", r"cheetah"), ("What do plants need from the sun?", r"light|energy"),
 ("At what temperature in Celsius does water freeze?", r"\b0\b|zero"), ("How many hours are in a day?", r"\b24\b|twenty.four"), ("What is a baby dog called?", r"pupp"),
 ("What is a baby cat called?", r"kitten"), ("What color do you get when you mix red and white?", r"pink"), ("What color do you get when you mix blue and yellow?", r"green"),
 ("Which season comes after winter?", r"spring"), ("What day comes after Monday?", r"tuesday"), ("What shape has three sides?", r"triangle"),
 ("How many sides does a square have?", r"four|\b4\b"), ("What is the closest planet to the Sun?", r"mercury"), ("Which planet is known as the red planet?", r"mars"),
 ("What is the largest country in the world?", r"russia"), ("Who painted the Mona Lisa?", r"vinci|leonardo"), ("What is the hardest natural substance?", r"diamond"),
 ("What do you call a scientist who studies stars?", r"astronom"), ("What bird cannot fly and lives in Antarctica?", r"penguin"), ("What instrument has black and white keys?", r"piano"),
 ("What is the main language of France?", r"french"), ("What country do kangaroos come from?", r"australia"), ("What is the name of the galaxy we live in?", r"milky way"),
 ("How many minutes are in an hour?", r"\b60\b|sixty"), ("What metal is attracted to magnets?", r"iron|steel|nickel"), ("At what temperature in Celsius does water boil?", r"\b100\b|hundred"),
 ("What organ pumps blood around the body?", r"heart"), ("What part of the body do we use to see?", r"eyes?\b"), ("What is the opposite of up?", r"down"),
 ("Which animal is known as the king of the jungle?", r"lion"), ("What sport uses an orange ball and a hoop?", r"basketball"), ("What is a group of wolves called?", r"pack"),
 ("How many wheels does a bicycle have?", r"two|\b2\b"), ("What fruit is yellow and curved?", r"banana"), ("What is the largest hot desert in the world?", r"sahara")]


def arithmetic(n=30, seed=11):
    rng = random.Random(seed); out = []
    for i in range(n):
        op = "+-x"[i % 3]
        if op == "+": a, b = rng.randint(2, 50), rng.randint(2, 50); v = a + b; q = f"What is {a} plus {b}?"
        elif op == "-": a, b = sorted((rng.randint(2, 60), rng.randint(2, 60)), reverse=True); v = a - b; q = f"What is {a} minus {b}?"
        else: a, b = rng.randint(2, 12), rng.randint(2, 12); v = a * b; q = f"What is {a} times {b}?"
        out.append((q, v))
    return out
