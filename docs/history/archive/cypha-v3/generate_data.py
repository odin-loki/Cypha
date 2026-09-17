"""
Cypha Multimodal Training Data — 500 verifiable examples
Modalities: math, logic, sequence, language, geography, science,
            array(synthetic image), binary pattern, question, sorting
All outputs are deterministic and checkable.
Run: python3 generate_training_data.py
Produces: cypha_train_500.txt  (input|||output format)
"""

import numpy as np
import random
import base64
import os

random.seed(42)
np.random.seed(42)

def arr_enc(a: np.ndarray) -> str:
    return "arr:" + base64.b64encode(a.astype(np.float32).tobytes()).decode()

# ── Synthetic 8×8 pixel grids (image modality) ──────────────────────

SHAPES = {
    "horizontal_line": lambda g: _set(g, rows=[4]),
    "vertical_line":   lambda g: _set(g, cols=[4]),
    "diagonal_fwd":    lambda g: _diag(g, False),
    "diagonal_bwd":    lambda g: _diag(g, True),
    "cross":           lambda g: _set(g, rows=[4], cols=[4]),
    "border":          lambda g: _set(g, rows=[0,7], cols=[0,7]),
    "top_half":        lambda g: _rect(g, 0,4, 0,8),
    "bottom_half":     lambda g: _rect(g, 4,8, 0,8),
    "left_half":       lambda g: _rect(g, 0,8, 0,4),
    "right_half":      lambda g: _rect(g, 0,8, 4,8),
    "center_square":   lambda g: _rect(g, 2,6, 2,6),
    "checkerboard":    lambda g: _checker(g),
    "dot_center":      lambda g: _rect(g, 3,5, 3,5),
    "x_pattern":       lambda g: _xpat(g),
    "corners":         lambda g: _corners(g),
}

def _set(g, rows=[], cols=[]):
    g = g.copy()
    for r in rows: g[r,:]=1.
    for c in cols: g[:,c]=1.
    return g

def _diag(g, rev):
    g = g.copy()
    for i in range(8):
        g[i, 7-i if rev else i] = 1.
    return g

def _rect(g, r0,r1,c0,c1):
    g = g.copy(); g[r0:r1, c0:c1] = 1.; return g

def _checker(g):
    g = g.copy()
    for i in range(8):
        for j in range(8):
            if (i+j)%2==0: g[i,j]=1.
    return g

def _xpat(g):
    g = g.copy()
    for i in range(8): g[i,i]=1.; g[i,7-i]=1.
    return g

def _corners(g):
    g = g.copy()
    for r,c in [(0,0),(0,7),(7,0),(7,7)]: g[r,c]=1.
    return g


# ── Numeric waveform modality ────────────────────────────────────────

def make_wave(kind: str, n=16) -> np.ndarray:
    t = np.linspace(0, 2*np.pi, n)
    if kind == "sine":       return np.sin(t)
    if kind == "cosine":     return np.cos(t)
    if kind == "sawtooth":   return (t/(2*np.pi)*2-1)
    if kind == "square":     return np.sign(np.sin(t))
    if kind == "triangle":   return 2*np.abs(t/np.pi - np.floor(t/np.pi + 0.5))
    if kind == "noise":      return np.random.randn(n)*0.5
    if kind == "ramp_up":    return np.linspace(0,1,n)
    if kind == "ramp_down":  return np.linspace(1,0,n)
    return np.zeros(n)


# ── Data generators ──────────────────────────────────────────────────

def gen_math(n=70):
    out = []
    for _ in range(n//4):
        a,b=random.randint(1,99),random.randint(1,99)
        out.append((f"{a}+{b}", str(a+b)))
    for _ in range(n//4):
        a,b=random.randint(10,99),random.randint(1,50)
        out.append((f"{a}-{b}", str(a-b)))
    for _ in range(n//4):
        a,b=random.randint(1,12),random.randint(1,12)
        out.append((f"{a}*{b}", str(a*b)))
    for _ in range(n//4):
        a,b=random.randint(2,9),random.randint(2,9)
        out.append((f"{a*b} / {a}", str(b)))
    for _ in range(n//8):
        a,b=random.randint(10,99),random.randint(2,9)
        out.append((f"{a} mod {b}", str(a%b)))
    return out[:n]

def gen_logic(n=50):
    out=[]
    for _ in range(n//3):
        a,b=random.randint(1,20),random.randint(1,20)
        out.append((f"is {a} > {b}", "true" if a>b else "false"))
    for _ in range(n//3):
        a,b=random.randint(1,20),random.randint(1,20)
        out.append((f"is {a} == {b}", "true" if a==b else "false"))
    for _ in range(n//6):
        out.append((f"not true",  "false"))
        out.append((f"not false", "true"))
    for _ in range(n//6):
        a=random.choice(["true","false"])
        b=random.choice(["true","false"])
        op=random.choice(["and","or"])
        av=(a=="true"); bv=(b=="true")
        rv=(av and bv) if op=="and" else (av or bv)
        res="true" if rv else "false"
        out.append((f"{a} {op} {b}", res))
    return out[:n]

def gen_sequences(n=60):
    out=[]
    for _ in range(n//4):  # arithmetic
        s,d=random.randint(1,10),random.randint(1,5)
        seq=[s+i*d for i in range(5)]
        out.append(("next: "+", ".join(map(str,seq)), str(seq[-1]+d)))
    for _ in range(n//4):  # geometric
        s,r=random.randint(1,3),random.randint(2,3)
        seq=[s*r**i for i in range(4)]
        out.append(("next: "+", ".join(map(str,seq)), str(seq[-1]*r)))
    for _ in range(n//4):  # sort
        nums=[random.randint(1,99) for _ in range(5)]
        out.append(("sort: "+" ".join(map(str,nums)),
                    " ".join(map(str,sorted(nums)))))
    for _ in range(n//4):  # squares
        s=random.randint(1,6)
        seq=[i**2 for i in range(s,s+4)]
        out.append(("next: "+", ".join(map(str,seq)), str((s+4)**2)))
    return out[:n]

def gen_language(n=80):
    animals = ["cat","dog","fox","wolf","bear","owl","tiger","lion",
               "horse","cow","sheep","duck","frog","snake","eagle"]
    sounds  = ["meow","bark","yip","howl","growl","hoot","roar","roar",
               "neigh","moo","baa","quack","croak","hiss","screech"]
    verbs   = ["run","jump","swim","fly","sleep","eat","write","sing","build","fall"]
    past    = ["ran","jumped","swam","flew","slept","ate","wrote","sang","built","fell"]
    adj     = ["big","small","fast","slow","hot","cold","hard","soft","bright","dark"]
    opp     = ["small","big","slow","fast","cold","hot","soft","hard","dark","bright"]
    out=[]
    for i in range(len(animals)):
        out.append((f"{animals[i]} sound", sounds[i]))
    for i in range(len(verbs)):
        out.append((f"{verbs[i]} past tense", past[i]))
    for i in range(len(adj)):
        out.append((f"opposite of {adj[i]}", opp[i]))
    random.shuffle(out)
    return out[:n]

def gen_geography(n=40):
    cities   = ["Paris","Tokyo","London","Berlin","Rome","Madrid",
                "Cairo","Seoul","Ottawa","Lima","Canberra","Brasilia",
                "Beijing","Moscow","Washington","Nairobi","Jakarta","Bangkok"]
    countries= ["France","Japan","UK","Germany","Italy","Spain",
                "Egypt","Korea","Canada","Peru","Australia","Brazil",
                "China","Russia","USA","Kenya","Indonesia","Thailand"]
    continents=["Europe","Asia","Europe","Europe","Europe","Europe",
                "Africa","Asia","North America","South America","Oceania",
                "South America","Asia","Europe","North America","Africa",
                "Asia","Asia"]
    out=[]
    for i in range(len(cities)):
        out.append((f"capital of {countries[i]}", cities[i]))
    for i in range(len(countries)):
        out.append((f"continent of {countries[i]}", continents[i]))
    random.shuffle(out)
    return out[:n]

def gen_science(n=30):
    return [
        ("boiling point of water celsius", "100"),
        ("freezing point of water celsius", "0"),
        ("speed of light units", "metres per second"),
        ("planets in solar system", "8"),
        ("atomic number of hydrogen", "1"),
        ("atomic number of carbon", "6"),
        ("atomic number of oxygen", "8"),
        ("lightest element", "hydrogen"),
        ("densest common metal", "osmium"),
        ("human chromosomes", "46"),
        ("bones in adult human body", "206"),
        ("chambers in human heart", "4"),
        ("chemical symbol for gold", "Au"),
        ("chemical symbol for iron", "Fe"),
        ("chemical symbol for water", "H2O"),
        ("chemical symbol for carbon dioxide", "CO2"),
        ("newton first law", "objects in motion stay in motion"),
        ("force equals", "mass times acceleration"),
        ("energy equals", "mass times speed of light squared"),
        ("absolute zero celsius", "-273"),
        ("ph of pure water", "7"),
        ("number of DNA bases", "4"),
        ("DNA bases", "ATCG"),
        ("cells in human body approximate", "37 trillion"),
        ("diameter of earth km", "12742"),
        ("distance earth to moon km", "384400"),
        ("gravitational constant symbol", "G"),
        ("planck constant symbol", "h"),
        ("avogadro number approximate", "6.022e23"),
        ("half life definition", "time for half of radioactive atoms to decay"),
    ][:n]

def gen_shapes(n=30):
    """Synthetic image modality — 8×8 pixel grids"""
    out=[]
    base=np.zeros((8,8),dtype=np.float32)
    shape_names=list(SHAPES.keys())
    for name in shape_names:
        grid=SHAPES[name](base)
        out.append((arr_enc(grid), name))
    # Add with noise
    for name in random.sample(shape_names, min(n-len(shape_names), len(shape_names))):
        grid=SHAPES[name](base)
        grid+=np.random.randn(8,8).astype(np.float32)*0.1
        out.append((arr_enc(grid), name))
    random.shuffle(out)
    return out[:n]

def gen_waveforms(n=20):
    """Numeric waveform modality"""
    out=[]
    kinds=["sine","cosine","sawtooth","square","triangle","ramp_up","ramp_down"]
    for kind in kinds:
        wave=make_wave(kind)
        out.append((arr_enc(wave), kind))
    # Variations
    for kind in random.sample(kinds, min(n-len(kinds), len(kinds))):
        wave=make_wave(kind)+np.random.randn(16).astype(np.float32)*0.05
        out.append((arr_enc(wave), kind))
    return out[:n]

def gen_binary_patterns(n=20):
    """Binary pattern modality"""
    out=[]
    patterns={
        "all_zeros":   bytes(64),
        "all_ones":    bytes([255]*64),
        "alternating": bytes([0,255]*32),
        "counting":    bytes(range(64)),
        "reverse":     bytes(range(63,-1,-1)),
        "high_byte":   bytes([128]*64),
        "random_low":  bytes([random.randint(0,63) for _ in range(64)]),
        "random_high": bytes([random.randint(192,255) for _ in range(64)]),
    }
    for name, data in patterns.items():
        encoded = "hex:" + data.hex()
        out.append((encoded, name))
    # Pad to n
    keys = list(patterns.keys())
    while len(out) < n:
        name = random.choice(keys)
        data = patterns[name]
        noise = bytes([(b + random.randint(-5,5)) % 256 for b in data])
        out.append(("hex:"+noise.hex(), name))
    return out[:n]

def gen_questions(n=20):
    return [
        ("what is the answer to life the universe and everything", "42"),
        ("how many sides does a triangle have", "3"),
        ("how many sides does a square have", "4"),
        ("how many sides does a hexagon have", "6"),
        ("how many days in a week", "7"),
        ("how many months in a year", "12"),
        ("how many hours in a day", "24"),
        ("how many minutes in an hour", "60"),
        ("how many seconds in a minute", "60"),
        ("how many degrees in a circle", "360"),
        ("how many degrees in a right angle", "90"),
        ("what comes after nine", "ten"),
        ("what comes before one", "zero"),
        ("largest planet in solar system", "Jupiter"),
        ("smallest planet in solar system", "Mercury"),
        ("closest star to earth", "Sun"),
        ("second closest star to earth", "Proxima Centauri"),
        ("largest ocean", "Pacific"),
        ("longest river", "Nile"),
        ("tallest mountain", "Everest"),
    ][:n]


# ── Assemble and write ───────────────────────────────────────────────

def generate(path="cypha_train_500.txt"):
    data = []
    data += gen_math(70)
    data += gen_logic(50)
    data += gen_sequences(60)
    data += gen_language(80)
    data += gen_geography(40)
    data += gen_science(30)
    data += gen_shapes(30)
    data += gen_waveforms(20)
    data += gen_binary_patterns(20)
    data += gen_questions(20)

    # Top up to 500 with extra math + language if short
    while len(data) < 500:
        data += gen_math(10)
    data = data[:500]

    random.shuffle(data)

    with open(path, 'w', encoding='utf-8') as f:
        for inp, out in data:
            f.write(f"{inp}|||{out}\n")

    # Print summary
    modalities = {
        'math':    sum(1 for i,_ in data if any(op in i for op in ['+','-','*','/','mod'])),
        'logic':   sum(1 for i,_ in data if i.startswith('is ') or i.startswith('not') or ' and ' in i or ' or ' in i),
        'sequence':sum(1 for i,_ in data if i.startswith('next:') or i.startswith('sort:')),
        'language':sum(1 for i,_ in data if 'sound' in i or 'past tense' in i or 'opposite' in i),
        'geo':     sum(1 for i,_ in data if 'capital' in i or 'continent' in i),
        'science': sum(1 for i,_ in data if any(w in i for w in ['boiling','atomic','chemical','newton','dna'])),
        'array':   sum(1 for i,_ in data if i.startswith('arr:')),
        'binary':  sum(1 for i,_ in data if i.startswith('hex:')),
        'question':sum(1 for i,_ in data if i.startswith('what') or i.startswith('how many') or i.startswith('largest') or i.startswith('closest')),
    }
    print(f"Generated {len(data)} examples → {path}")
    print(f"Modality breakdown:")
    for m,n in sorted(modalities.items(), key=lambda x:-x[1]):
        print(f"  {m:12} {n:4}")
    print(f"\nVerification samples:")
    samples = [d for d in data if not d[0].startswith('arr:') and not d[0].startswith('hex:')][:5]
    for i,o in samples:
        print(f"  '{i}' → '{o}'")
    array_samples = [d for d in data if d[0].startswith('arr:')][:2]
    for i,o in array_samples:
        print(f"  arr:[{len(i)} chars] → '{o}'")

    return data

if __name__ == "__main__":
    generate()
