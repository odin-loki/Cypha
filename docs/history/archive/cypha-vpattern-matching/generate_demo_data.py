"""
Generate demonstration data for Cypha HRNA showcase

Creates diverse tasks to demonstrate system capabilities:
- Arithmetic
- Language (sounds, synonyms, past tense)
- Geography (capitals, countries)
- Logic (comparisons, boolean)
- Sorting
- General knowledge
"""

import random

def generate_demo_data(n_samples=1000):
    """Generate diverse training data"""
    
    verbs = ["run", "jump", "swim", "fly", "crawl", "sleep", "eat", "read", "write"]
    past_tense = ["ran", "jumped", "swam", "flew", "crawled", "slept", "ate", "read", "wrote"]
    
    animals = ["cat", "dog", "fox", "wolf", "bear", "owl", "tiger", "mouse", "rat", "horse"]
    sounds = ["meow", "bark", "chirp", "howl", "growl", "hoot", "roar", "squeak", "squeak", "neigh"]
    
    cities = ["Paris", "London", "Tokyo", "New York", "Beijing", "Sydney", "Cairo", "Berlin", "Rome", "Seoul"]
    countries = ["France", "UK", "Japan", "USA", "China", "Australia", "Egypt", "Germany", "Italy", "Korea"]
    
    data = []
    
    for i in range(n_samples):
        task = random.randint(0, 9)
        
        if task == 0:
            # Addition
            a, b = random.randint(1, 199), random.randint(1, 199)
            data.append(f"{a}+{b}|||{a+b}")
        
        elif task == 1:
            # Animal sounds
            idx = random.randint(0, len(animals)-1)
            data.append(f"{animals[idx]} sound|||{sounds[idx]}")
        
        elif task == 2:
            # Capitals
            idx = random.randint(0, len(countries)-1)
            data.append(f"capital of {countries[idx]}|||{cities[idx]}")
        
        elif task == 3:
            # Past tense
            idx = random.randint(0, len(verbs)-1)
            data.append(f"{verbs[idx]} past|||{past_tense[idx]}")
        
        elif task == 4:
            # Sorting
            nums = [random.randint(1, 99) for _ in range(5)]
            sorted_nums = sorted(nums)
            nums_str = " ".join(map(str, nums))
            sorted_str = " ".join(map(str, sorted_nums))
            data.append(f"sort: {nums_str}|||{sorted_str}")
        
        elif task == 5:
            # Comparisons
            a, b = random.randint(1, 20), random.randint(1, 20)
            result = "true" if a > b else "false"
            data.append(f"is {a} > {b}|||{result}")
        
        elif task == 6:
            # Synonyms (random pairs)
            word1 = random.choice(animals + verbs)
            word2 = random.choice(animals + verbs)
            data.append(f"{word1} synonym|||{word2}")
        
        elif task == 7:
            # The answer
            data.append(f"What is the answer to life, the universe and everything?|||42")
        
        elif task == 8:
            # Colors (fictional)
            colors = ["red", "blue", "green", "yellow", "violet", "cyan", "orange", "pink"]
            color = random.choice(colors)
            hex_code = "#" + "".join(random.choices("0123456789ABCDEF", k=6))
            data.append(f"{color} in HEX|||{hex_code}")
        
        else:
            # Questions
            animal = random.choice(animals)
            verb = random.choice(verbs)
            answer = random.choice(["yes", "no", "true", "false"])
            data.append(f"{animal} {verb}?|||{answer}")
    
    return data

def create_showcase_datasets():
    """Create multiple dataset sizes for demonstration"""
    
    print("Generating demonstration datasets...")
    
    # Small dataset for quick demos
    small_data = generate_demo_data(100)
    with open("demo_small.txt", "w") as f:
        for line in small_data:
            f.write(line + "\n")
    print(f"✓ Created demo_small.txt ({len(small_data)} examples)")
    
    # Medium dataset for training
    medium_data = generate_demo_data(1000)
    with open("demo_medium.txt", "w") as f:
        for line in medium_data:
            f.write(line + "\n")
    print(f"✓ Created demo_medium.txt ({len(medium_data)} examples)")
    
    # Large dataset for scaling test
    large_data = generate_demo_data(10000)
    with open("demo_large.txt", "w") as f:
        for line in large_data:
            f.write(line + "\n")
    print(f"✓ Created demo_large.txt ({len(large_data)} examples)")
    
    # Create a curated test set
    test_data = [
        # Math
        "12+165|||177",
        "25+37|||62",
        "100+50|||150",
        
        # Animal sounds
        "cat sound|||meow",
        "dog sound|||bark",
        "owl sound|||hoot",
        
        # Geography
        "capital of France|||Paris",
        "capital of Japan|||Tokyo",
        "capital of USA|||New York",
        
        # Logic
        "is 5 > 3|||true",
        "is 2 > 10|||false",
        
        # Sorting
        "sort: 5 2 9 1|||1 2 5 9",
        "sort: 8 3 6 1|||1 3 6 8",
        
        # Past tense
        "run past|||ran",
        "jump past|||jumped",
        
        # General
        "What is the answer to life, the universe and everything?|||42",
    ]
    
    with open("demo_test.txt", "w") as f:
        for line in test_data:
            f.write(line + "\n")
    print(f"✓ Created demo_test.txt ({len(test_data)} curated examples)")
    
    print("\nDatasets ready for showcase!")
    print("\nUsage:")
    print("  Quick demo:   python showcase_demo.py")
    print("  Train small:  cypha.train('demo_small.txt', epochs=3)")
    print("  Train medium: cypha.train('demo_medium.txt', epochs=1)")
    print("  Train large:  cypha.train('demo_large.txt', epochs=1)")

if __name__ == "__main__":
    create_showcase_datasets()
