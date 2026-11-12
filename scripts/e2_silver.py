#!/usr/bin/env python3
"""E2 silver entity labels from Wikipedia link anchors (docs/reports/E2_ENTITIES.md).

    python3 -I scripts/e2_silver.py ENWIK9 SKIP_OFFSETS.txt OUTDIR [--train-sents N] [--dev-sents N] [--test-sents N]

Pass 1 types every page title from the dump's own markup: its infobox name and its categories
(PERSON, PLACE, ORG; any other article is MISC) and follows redirects. Pages whose text starts in
enwik8 bytes [16 MiB, 1e8) (SCRIBE's [16, 72) MiB and GEN-A's [75497472, 1e8)) are not read at all.

Pass 2 reads prose sentences only from enwik9 [1e8, 1e9), skipping the overlap pages (SKIP_OFFSETS,
the same list as GENERATION_PLAN 13.38). A link [[T|anchor]] whose anchor starts with a capital
gives a span typed by T. Year and day links ([[1954]], [[March 5]]) and plain year, day and month
expressions give DATE; other numerals give NUMBER. Inside a page, an anchor once typed (and the
page's own title, and a person's surname) labels its later unlinked mentions. A sentence is kept
only if every capitalised word after its first lies inside a span, so unlinked names do not become
silent "O" labels.

Split by page: md5(title) mod 20 == 0 -> test, == 1 -> dev, else train.

OUTDIR/hand_pool.tsv: 200 prose sentences of the test pages drawn without the coverage filter (seeded), for the
hand-checked set.

Output (OUTDIR/{train,dev,test}.tsv): page title, sentence text, spans "start:end:TYPE;..." in UTF-8
byte offsets. OUTDIR/types.tsv: title -> type (pass 1). OUTDIR/stats.txt: counts.

Author: Odin Loch
"""
import hashlib
import html
import re
import sys
import collections

LO_EXCL, HI_EXCL = 16 * 1024 * 1024, 100_000_000
MONTHS = ("January February March April May June July August September October November December").split()
MONTH_RE = "(?:" + "|".join(MONTHS) + ")"

PERSON_INFOBOX = re.compile(r"(biography|person|officeholder|pope|monarch|senator|governor|president|prime minister|"
                            r"scientist|philosopher|writer|artist|actor|actress|cricketer|football ?biography|nfl ?player|"
                            r"baseball ?player|military person|royalty|saint|chancellor|judge|astronaut|economist|"
                            r"comedian|model|wrestler|boxer|tennis|golfer|swimmer|f1 driver|mlb player|nba player|"
                            r"hrhstyles|popestyles|congressman|representative|mp|minister|celebrity|chef|architect)\b")
MUSIC_ARTIST = re.compile(r"musical artist|band")
PLACE_INFOBOX = re.compile(r"(city|town|village|country|settlement|province|county|counties|state|region|river|lake|"
                           r"mountain|island|place|municipality|commune|department|district|prefecture|canton|borough|"
                           r"oblast|kreis|gemeinde|suburb|neighbourhood|neighborhood|capital|ocean|sea|desert|volcano|"
                           r"continent|islands|archipelago|peninsula|protected area|park|uk place|england place|"
                           r"scotland place|ireland place|wales place|australian|canadian|indian|japan|german|"
                           r"dutch province|lga|territory|parish)\b")
ORG_INFOBOX = re.compile(r"(company|corporation|organization|organisation|university|college|school|political party|"
                         r"football club|club|team|airline|newspaper|record label|band|union|agency|government agency|"
                         r"institute|museum|broadcaster|tv channel|television network|radio station|network|"
                         r"magazine|publisher|bank|non-profit|ngo|military unit|army|navy|church|religious order|"
                         r"sports league|league|federation|association|stock exchange|intergovernmental)\b")
PERSON_CAT = re.compile(r"(^\d+s?( bc)? (births|deaths)$|^living people$|^people from |^people of |"
                        r"(^| )(people|players|actors|actresses|singers|writers|novelists|poets|politicians|musicians|"
                        r"composers|painters|artists|scientists|physicists|chemists|mathematicians|philosophers|"
                        r"economists|historians|journalists|footballers|cricketers|monarchs|kings|queens|emperors|"
                        r"popes|saints|presidents|prime ministers|generals|songwriters|guitarists|pianists|"
                        r"directors|producers|engineers|inventors|astronomers|biologists|explorers|bishops|"
                        r"cardinals|theologians|astronauts|lawyers|judges|businesspeople|comedians|dancers|"
                        r"athletes|boxers|wrestlers|swimmers|cyclists|drivers|rappers|bassists|drummers|sculptors|"
                        r"architects|photographers|designers|illustrators|cartoonists|anarchists|activists|"
                        r"socialists|communists|nobility|knights|dukes|earls|freemasons|novelists|playwrights|"
                        r"screenwriters|dramatists|essayists|critics|linguists|psychologists|sociologists|"
                        r"anthropologists|geologists|botanists|zoologists|physicians|surgeons|nurses|soldiers|"
                        r"admirals|officers|pilots|aviators|missionaries|rabbis|imams|monks|nuns|priests|"
                        r"martyrs|pharaohs|sultans|caliphs|shahs|tsars|czars|princes|princesses|consorts|"
                        r"senators|governors|mayors|diplomats|ambassadors|spies|criminals|murderers|"
                        r"entrepreneurs|bankers|industrialists|philanthropists|chess players|tennis players|"
                        r"golfers|jockeys|coaches|managers|referees|olympians|medalists|laureates|"
                        r"citizens of [a-z ]+|emigrants to [a-z ]+|immigrants to [a-z ]+))$|"
                        r"^(american|english|british|french|german|italian|russian|spanish|irish|scottish|welsh|"
                        r"canadian|australian|japanese|chinese|indian|dutch|swedish|polish|greek|roman|"
                        r"ancient [a-z]+|african american) [a-z ]+$")
PLACE_CAT = re.compile(r"(^| )(cities|towns|villages|capitals|countries|counties|provinces|states|regions|rivers|"
                       r"mountains|islands|lakes|populated places|municipalities|boroughs|districts|departments|"
                       r"prefectures|cantons|communes|seas|oceans|deserts|volcanoes|archipelagos|peninsulas|"
                       r"continents|neighbourhoods|neighborhoods|suburbs|member states|wto members|"
                       r"settlements|hamlets|parishes|territories|oblasts|republics|kingdoms|empires|"
                       r"former countries|landlocked countries|island countries|dependencies|colonies|"
                       r"cities and towns|unitary authorities|census-designated places|ghost towns|"
                       r"geography of [a-z ]+ stubs|valleys|bays|straits|gulfs|capes|glaciers|plateaus|"
                       r"mountain ranges|regions of [a-z ]+)( |$)")
ORG_CAT = re.compile(r"(^| )(companies|company|organizations|organisations|universities|colleges|schools|"
                     r"political parties|parties|bands|musical groups|football clubs|clubs|teams|airlines|"
                     r"agencies|newspapers|record labels|manufacturers|banks|institutes|societies|federations|"
                     r"associations|councils|ministries|armies|navies|air forces|military units|charities|"
                     r"brands|publishers|broadcasters|television networks|radio stations|stock exchanges|"
                     r"international organizations|religious orders|orders|museums|libraries|leagues|"
                     r"conglomerates|corporations|cooperatives|think tanks|trade unions|unions|"
                     r"intelligence agencies|police|government agencies|new deal agency|legislatures|"
                     r"parliaments|courts|churches|denominations|monasteries|abbeys|dynasties|houses)( |$)")
NOT_TYPE_CAT = re.compile(r"^(lists? of|history of|albums|songs|films|novels|books|works|timelines?|"
                          r"wikipedia|articles|stubs|disambiguation)")


def norm_title(t):
    t = html.unescape(t).replace("_", " ").strip()
    t = re.sub(r"\s+", " ", t)
    if "#" in t:
        t = t.split("#", 1)[0].strip()
    return t[:1].upper() + t[1:] if t else t


def classify(infobox, cats):
    ib = infobox.lower().replace("_", " ").strip() if infobox else ""
    lc = [c.lower().strip() for c in cats]
    per_strong = any(re.match(r"^\d+s?( bc)? (births|deaths)$", c) or c == "living people" for c in lc)
    if per_strong:
        return "PERSON"
    if ib:
        if MUSIC_ARTIST.search(ib):
            # musical artist: a person or a band; the categories decide, a band is the default
            v = vote(lc)
            return "PERSON" if v == "PERSON" else "ORG"
        if PERSON_INFOBOX.search(ib):
            return "PERSON"
        if ORG_INFOBOX.search(ib):
            return "ORG"
        if PLACE_INFOBOX.search(ib):
            return "PLACE"
    v = vote(lc)
    return v if v else "MISC"


def vote(lc):
    n = {"PERSON": 0, "PLACE": 0, "ORG": 0}
    for c in lc:
        if NOT_TYPE_CAT.match(c):
            continue
        if PERSON_CAT.search(c):
            n["PERSON"] += 1
        if PLACE_CAT.search(c):
            n["PLACE"] += 1
        if ORG_CAT.search(c):
            n["ORG"] += 1
    best = max(n.values())
    if best == 0:
        return None
    top = [k for k, v in n.items() if v == best]
    return top[0] if len(top) == 1 else None


PAGE_RE = re.compile(rb"<title>(.*?)</title>.*?<text xml:space=\"preserve\">(.*?)</text>", re.S)
REDIR_RE = re.compile(rb"^\s*#redirect\s*:?\s*\[\[([^\]|#]+)", re.I)
CAT_RE = re.compile(rb"\[\[\s*Category\s*:\s*([^\]|]+)", re.I)
IB_RE = re.compile(rb"\{\{\s*[Ii]nfobox[ _]*([^|}<\n]*)")
IB2_RE = re.compile(rb"\{\{\s*([A-Za-z _]*(?:[Bb]iography|[Cc]ountry|[Cc]ity|[Cc]ompany|[Bb]and|[Uu]niversity|"
                    rb"[Pp]olitical [Pp]arty|[Ff]ootball club|[Ss]ettlement|[Mm]usical artist))\s*[|\n]")
NONMAIN = re.compile(r"^(Wikipedia|Category|Template|Image|File|Portal|Help|MediaWiki|User|Talk|Special|Media|"
                     r"[A-Za-z ]+ talk|WP|Project):")


def pass1(data):
    types, redirect = {}, {}
    n = 0
    for m in PAGE_RE.finditer(data):
        off = m.start(2)
        if LO_EXCL <= off < HI_EXCL:
            continue
        title = norm_title(m.group(1).decode("utf-8", "replace"))
        if NONMAIN.match(title):
            continue
        body = m.group(2)
        r = REDIR_RE.match(body)
        if r:
            redirect[title] = norm_title(r.group(1).decode("utf-8", "replace"))
            continue
        cats = [c.decode("utf-8", "replace") for c in CAT_RE.findall(body)]
        if any("disambiguation" in c.lower() for c in cats) or re.search(rb"\{\{\s*(disambig|dab|geodis|hndis)", body, re.I):
            types[title] = "DISAMBIG"
            continue
        ib = IB_RE.search(body) or IB2_RE.search(body)
        types[title] = classify(ib.group(1).decode("utf-8", "replace") if ib else "", cats)
        n += 1
    return types, redirect


def resolve(t, types, redirect):
    for _ in range(3):
        if t in types:
            return t
        if t in redirect:
            t = redirect[t]
        else:
            return None
    return t if t in types else None


# ---------------------------------------------------------------- markup stripping
def remove_nested(s, open_, close):
    out, depth, i = [], 0, 0
    lo, lc = len(open_), len(close)
    while i < len(s):
        if s.startswith(open_, i):
            depth += 1
            i += lo
        elif depth and s.startswith(close, i):
            depth -= 1
            i += lc
        else:
            if not depth:
                out.append(s[i])
            i += 1
    return "".join(out)


def remove_prefixed_links(s):
    # [[Image:...]], [[File:...]], [[Category:...]], interwiki [[xx:...]]: removed with any nested links.
    out, i = [], 0
    pat = re.compile(r"\[\[\s*([A-Za-z\-]+)\s*:")
    while True:
        m = pat.search(s, i)
        if not m:
            out.append(s[i:])
            break
        out.append(s[i:m.start()])
        depth, j = 0, m.start()
        while j < len(s):
            if s.startswith("[[", j):
                depth += 1
                j += 2
            elif s.startswith("]]", j):
                depth -= 1
                j += 2
                if depth == 0:
                    break
            else:
                j += 1
        i = j
    return "".join(out)


def clean_body(raw):
    s = html.unescape(raw).replace("\xa0", " ")
    s = re.sub(r"<!--.*?-->", "", s, flags=re.S)
    s = re.sub(r"<ref[^>/]*/>", "", s)
    s = re.sub(r"<ref[^>]*>.*?</ref>", "", s, flags=re.S)
    s = re.sub(r"<(math|gallery|timeline|table|pre|code|source|nowiki|div|span)[^>]*>.*?</\1>", "", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", "", s)
    s = remove_nested(s, "{{", "}}")
    s = remove_nested(s, "{|", "|}")
    s = remove_prefixed_links(s)
    return s


LINK_RE = re.compile(r"\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\]([a-z]*)")
EXT_RE = re.compile(r"\[(?:https?|ftp)://[^\s\]]+(?: ([^\]]*))?\]")
YEAR_LINK = re.compile(r"^\d{1,4}( BC| AD| BCE| CE)?$")
DECADE_LINK = re.compile(r"^\d{1,3}0s( BC)?$")
CENTURY_LINK = re.compile(r"^\d{1,2}(st|nd|rd|th) century( BC)?$")
DAY_LINK = re.compile(r"^(" + MONTH_RE + r" \d{1,2}|\d{1,2} " + MONTH_RE + r")$")
MONTH_YEAR_LINK = re.compile(r"^" + MONTH_RE + r" \d{4}$")
DEMONYM_END = re.compile(r"(ese|ian|ish|ch|an|i|ic|ean|er)$")


def link_type(target, anchor, types, redirect):
    t = target.strip()
    if YEAR_LINK.match(t) or DECADE_LINK.match(t) or CENTURY_LINK.match(t) or DAY_LINK.match(t) or MONTH_YEAR_LINK.match(t):
        return "DATE"
    if not anchor[:1].isupper():
        return None  # a common-noun link: not an entity
    if t[:1].islower() or NONMAIN.match(norm_title(t)) or ":" in t[:3]:
        return "?"
    r = resolve(norm_title(t), types, redirect)
    if r is None:
        return "?"  # no page: the anchor cannot be typed
    ty = types[r]
    if ty == "DISAMBIG":
        return "?"
    if ty == "PLACE":
        at, tt = anchor.split(), r.split()
        if len(at) == 1 and at[0] not in tt and DEMONYM_END.search(at[0]):
            return "MISC"  # [[France|French]]: a demonym, not the place
    return ty


def render_line(line, types, redirect):
    """Plain text of a prose line and its typed link spans (character offsets). '?' marks an untypable name."""
    line = EXT_RE.sub(lambda m: m.group(1) or "", line)
    out, spans, i = [], [], 0
    pos = 0
    for m in LINK_RE.finditer(line):
        seg = line[i:m.start()]
        out.append(seg)
        pos += len(seg)
        target, anchor, trail = m.group(1), m.group(2), m.group(3)
        if anchor is None or anchor.strip() == "":
            anchor = target.split("#")[0] if anchor is None else re.sub(r"\s*\(.*\)$", "", target)
        anchor = anchor.strip()
        text = anchor + trail
        ty = link_type(target, anchor, types, redirect)
        if ty:
            spans.append([pos, pos + len(text), ty, norm_title(target)])
        out.append(text)
        pos += len(text)
        i = m.end()
    out.append(line[i:])
    plain = "".join(out)
    # bold and italic quotes: remove and shift the spans
    res, shift, k = [], [0] * (len(plain) + 1), 0
    j = 0
    while j < len(plain):
        if plain.startswith("''", j):
            n = 2
            while j + n < len(plain) and plain[j + n] == "'":
                n += 1
            for q in range(j, j + n):
                shift[q] = k
            j += n
            continue
        shift[j] = k
        res.append(plain[j])
        k += 1
        j += 1
    shift[len(plain)] = k
    plain2 = "".join(res)
    sp2 = [[shift[a], shift[b], t, tg] for a, b, t, tg in spans]
    sp2 = [s for s in sp2 if s[1] > s[0]]
    return plain2, sp2


ABBREV = set("Mr Mrs Ms Dr St Jr Sr Prof Gen Col Lt Sgt Capt Rev Mt Ft vs etc Inc Ltd Co Corp No Vol pp ca c Jan Feb Mar Apr Jun Jul Aug Sep Sept Oct Nov Dec".split())


def split_sentences(text):
    """Character ranges of sentences: end at . ! ? followed by space and an upper-case letter, a quote or a digit."""
    out, a = [], 0
    for m in re.finditer(r"[.!?][\"')]?\s+(?=[A-Z\"'(0-9])", text):
        e = m.start() + 1
        word = re.search(r"(\S+)$", text[a:m.start()])
        w = word.group(1) if word else ""
        if w in ABBREV or re.fullmatch(r"[A-Z]", w) or re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", w):
            continue
        end = m.end() - len(m.group(0)) + len(m.group(0).rstrip())
        out.append((a, end))
        a = m.end()
    if a < len(text):
        out.append((a, len(text)))
    return out


NUM_RE = re.compile(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")
DATE_RES = [
    re.compile(r"\b" + MONTH_RE + r" \d{1,2}(?:st|nd|rd|th)?,? \d{3,4}\b"),
    re.compile(r"\b\d{1,2} " + MONTH_RE + r",? \d{3,4}\b"),
    re.compile(r"\b" + MONTH_RE + r" \d{1,2}(?:st|nd|rd|th)?\b"),
    re.compile(r"\b\d{1,2} " + MONTH_RE + r"\b"),
    re.compile(r"\b" + MONTH_RE + r",? \d{4}\b"),
    re.compile(r"\b\d{3}0s\b"),
    re.compile(r"\b\d{1,4} (?:BC|AD|BCE|CE)\b"),
    re.compile(r"\b(?:1[0-9]{3}|20[0-2][0-9])\b(?![,.]\d)"),
    re.compile(r"\b(?:in|In|of|since|until|by|from|early|late|during|between|and|to|By|Since|Until|From|During) (" + MONTH_RE + r")\b"),
]


def overlaps(spans, a, b):
    return any(not (b <= s[0] or a >= s[1]) for s in spans)


def add_rule_spans(text, spans):
    """DATE and NUMBER spans outside the link spans (silver rules)."""
    for rx in DATE_RES:
        for m in rx.finditer(text):
            g = 1 if rx.groups else 0
            if not overlaps(spans, m.start(g), m.end(g)):
                spans.append([m.start(g), m.end(g), "DATE", ""])
    for m in NUM_RE.finditer(text):
        if not overlaps(spans, m.start(), m.end()):
            spans.append([m.start(), m.end(), "NUMBER", ""])
    spans.sort()
    # merge DATE spans joined by ", " or " " ("[[March 5]], [[1954]]")
    merged = []
    for s in spans:
        if merged and merged[-1][2] == "DATE" and s[2] == "DATE" and text[merged[-1][1]:s[0]] in (", ", " ", ","):
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def add_propagated(text, spans, known):
    """Later unlinked mentions of names typed earlier in the page (longest match, whole words; "Kyrgyzstan's" matches)."""
    if not known:
        return spans
    toks = [(m.start(), m.end()) for m in re.finditer(r"[\w.\-&]+", text)]
    i = 0
    while i < len(toks):
        a = toks[i][0]
        if not text[a].isupper() or overlaps(spans, a, toks[i][1]):
            i += 1
            continue
        took = 0
        for n in range(min(6, len(toks) - i), 0, -1):
            j = i + n - 1
            if any(text[toks[k][1]:toks[k + 1][0]] != " " for k in range(i, j)):
                continue
            b = toks[j][1]
            cand = text[a:b]
            ty = known.get(cand) or (known.get(cand[:-1]) if cand.endswith(".") else None)
            if ty:
                if cand.endswith(".") and cand not in known:
                    b -= 1
                if not overlaps(spans, a, b):
                    spans.append([a, b, ty, ""])
                    took = n
                break
        i += took if took else 1
    spans.sort()
    return spans


def full_coverage(text, spans, common):
    """Every capitalised word lies inside a typed span (and no span is '?'). The sentence's first word may stay
    outside a span only if it is a common lowercase word of the corpus (common)."""
    if any(s[2] == "?" for s in spans):
        return False
    first = True
    for m in re.finditer(r"[\w\-.&]+", text):
        w = m.group(0)
        if first:
            first = False
            if w[:1].isupper() and not overlaps(spans, m.start(), m.end()) and w.lower().rstrip(".") not in common:
                return False
            continue
        if w[:1].isupper() and w != "I":
            if not overlaps(spans, m.start(), m.end()):
                return False
    return True


def bad_sentence(text):
    if re.search(r"[\[\]{}|=<>]|https?:|&#?[a-z0-9]+;|&#", text):
        return True
    if not re.search(r"[.!?][\"')]?$", text):
        return True
    n = len(text.split())
    return n < 6 or n > 60 or not text[:1].isalnum() and text[:1] not in "\"'("


def page_split(title):
    h = int(hashlib.md5(title.encode("utf-8")).hexdigest(), 16) % 20
    return "test" if h == 0 else "dev" if h == 1 else "train"


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    path, skip_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
    quota = {"train": 120000, "dev": 3000, "test": 3000}
    args = sys.argv[4:]
    for k in range(0, len(args), 2):
        quota[args[k].lstrip("-").split("-")[0]] = int(args[k + 1])
    skip = set(int(x) for x in open(skip_path) if x.strip().isdigit())
    with open(path, "rb") as f:
        data = f.read()
    types, redirect = pass1(data)
    # common lowercase words (for the sentence's first word): seen lowercase at least 5 times in enwik9 [1e8, 1.5e8)
    common = collections.Counter(re.findall(r"(?<![\w'])[a-z][a-z'\-]*", data[100_000_000:150_000_000].decode("utf-8", "replace")))
    common = {w for w, c in common.items() if c >= 5}
    tc = collections.Counter(types.values())
    with open(out + "/types.tsv", "w", encoding="utf-8") as f:
        for t, ty in sorted(types.items()):
            f.write(f"{t}\t{ty}\n")
    files = {k: open(f"{out}/{k}.tsv", "w", encoding="utf-8") for k in quota}
    got = {k: 0 for k in quota}
    toks = {k: 0 for k in quota}
    span_n = {k: collections.Counter() for k in quota}
    pages = collections.Counter()
    seen_sent = 0
    pool = []
    for m in PAGE_RE.finditer(data, pos=100_000_000):
        off = m.start(2)
        if off < 100_000_000 or off in skip:
            continue
        title = norm_title(m.group(1).decode("utf-8", "replace"))
        if NONMAIN.match(title) or title not in types or types[title] == "DISAMBIG":
            continue
        sp = page_split(title)
        if got[sp] >= quota[sp]:
            if all(got[k] >= quota[k] for k in quota):
                break
            continue
        pages[sp] += 1
        body = clean_body(m.group(2).decode("utf-8", "replace"))
        known = {}
        pty = types[title]
        base = re.sub(r"\s*\(.*\)$", "", title)
        if pty in ("PERSON", "PLACE", "ORG", "MISC") and base[:1].isupper():
            known[base] = pty
            if pty == "PERSON" and len(base.split()) >= 2 and len(base.split()[-1]) > 2:
                known.setdefault(base.split()[-1], "PERSON")
        for line in body.split("\n"):
            line = line.strip()
            if not line or line[0] in "=*#:;|!{}<" or line.startswith("----"):
                continue
            plain, spans = render_line(line, types, redirect)
            # bold lead: the page's own name (render_line removed the quotes; propagation labels it)
            for a, b, ty, tg in spans:
                if ty in ("PERSON", "PLACE", "ORG", "MISC"):
                    nm = plain[a:b]
                    known.setdefault(nm, ty)
                    if ty == "PERSON" and len(nm.split()) >= 2 and len(nm.split()[-1]) > 2:
                        known.setdefault(nm.split()[-1], "PERSON")
            for a, b in split_sentences(plain):
                raw = plain[a:b]
                lead = len(raw) - len(raw.lstrip())
                s_text = raw.strip()
                if not s_text:
                    continue
                a2 = a + lead
                b2 = a2 + len(s_text)
                seen_sent += 1
                if bad_sentence(s_text):
                    continue
                ss = []
                okk = True
                for x, y, ty, tg in spans:
                    if x >= a2 and y <= b2:
                        ss.append([x - a2, y - a2, ty, tg])
                    elif not (y <= a2 or x >= b2):
                        okk = False
                if not okk:
                    continue
                ss = add_propagated(s_text, ss, known)
                ss = add_rule_spans(s_text, ss)
                if sp == "test" and got[sp] < quota[sp]:
                    pool.append((title, s_text))  # the hand-check pool: test pages, no coverage filter
                if not full_coverage(s_text, ss, common):
                    continue
                if got[sp] >= quota[sp]:
                    break
                enc = s_text.encode("utf-8")
                bspans = []
                for x, y, ty, tg in ss:
                    bx, by = len(s_text[:x].encode("utf-8")), len(s_text[:y].encode("utf-8"))
                    bspans.append(f"{bx}:{by}:{ty}")
                    span_n[sp][ty] += 1
                files[sp].write(f"{title}\t{s_text}\t{';'.join(bspans)}\n")
                got[sp] += 1
                toks[sp] += len(enc.split())
    for f in files.values():
        f.close()
    import random
    random.Random(20261112).shuffle(pool)
    with open(out + "/hand_pool.tsv", "w", encoding="utf-8") as f:
        for t, x in pool[:200]:
            f.write(f"{t}\t{x}\n")
    with open(out + "/stats.txt", "w") as f:
        f.write(f"typed titles {len(types)}  redirects {len(redirect)}  types {dict(tc)}\n")
        f.write(f"sentences read {seen_sent}\n")
        for k in quota:
            f.write(f"{k}: pages {pages[k]} sentences {got[k]} words {toks[k]} spans {dict(span_n[k])}\n")
    print(open(out + "/stats.txt").read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
