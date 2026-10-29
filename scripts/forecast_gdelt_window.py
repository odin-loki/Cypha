#!/usr/bin/env python3
"""Fetch GDELT 1.0 daily event exports for a date window and convert them to the
forecast CSV (year,month,day,actor1,actor2,cameo_code,goldstein,theater).

Same theater mapping as scripts/fetch_forecast_data.ps1 (Convert-GdeltExportLine),
with one fix: cameo_code is EventBaseCode (3 digits), not EventCode, so the CAMEO
root is cameo_code / 10 for every row (EventCode "1823" would give root 182).
Keeps the first --max-per-day converted rows of each day, in file order.

Usage: forecast_gdelt_window.py OUT_DIR YYYYMMDD [YYYYMMDD ...] [--max-per-day N]
Writes OUT_DIR/gdelt_YYYYMMDD.csv (UTF-8) and OUT_DIR/manifest.tsv (date, url,
zip sha256, rows kept). Zips are deleted after conversion.
"""
import hashlib
import io
import os
import re
import sys
import urllib.request
import zipfile

URL = "https://data.gdeltproject.org/events/{d}.export.CSV.zip"
PATTERNS = [
    ("TWN", re.compile(r"CHN|TWN|CHINA|TAIWAN")),
    ("UKR", re.compile(r"RUS|UKR|RUSSIA|UKRAINE")),
    ("MID", re.compile(r"ISR|IRN|PSE|ISRAEL|IRAN|PALEST")),
    ("PRK", re.compile(r"PRK|KOR|KOREA")),
    ("SAS", re.compile(r"IND|PAK|INDIA|PAKIST")),
]


def convert(line):
    f = line.rstrip("\r\n").split("\t")
    if len(f) < 31:
        return None
    sql = f[1]
    if len(sql) < 8:
        return None
    a1 = f[6].replace('"', "").strip()
    a2 = f[16].replace('"', "").strip()
    base = f[27].replace('"', "").strip()
    gold = f[30].replace('"', "").strip()
    if not a1 or not a2 or not base.isdigit():
        return None
    u = (a1 + a2).upper()
    theater = "GLB"
    for name, pat in PATTERNS:
        if pat.search(u):
            theater = name
            break
    a1 = a1.replace(",", " ")
    a2 = a2.replace(",", " ")
    return f"{sql[0:4]},{int(sql[4:6])},{int(sql[6:8])},{a1},{a2},{int(base)},{gold},{theater}"


def main(argv):
    if len(argv) < 3:
        print(__doc__)
        return 2
    out = argv[1]
    max_per_day = 0
    dates = []
    i = 2
    while i < len(argv):
        if argv[i] == "--max-per-day":
            max_per_day = int(argv[i + 1])
            i += 2
            continue
        dates.append(argv[i])
        i += 1
    os.makedirs(out, exist_ok=True)
    man = open(os.path.join(out, "manifest.tsv"), "a", encoding="utf-8")
    for d in dates:
        url = URL.format(d=d)
        req = urllib.request.Request(url, headers={"User-Agent": "cypha-forecast-research/1.0"})
        data = urllib.request.urlopen(req, timeout=180).read()
        sha = hashlib.sha256(data).hexdigest()
        rows = ["year,month,day,actor1,actor2,cameo_code,goldstein,theater"]
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = z.namelist()[0]
            with z.open(name) as fh:
                for raw in io.TextIOWrapper(fh, encoding="utf-8", errors="replace"):
                    c = convert(raw)
                    if c is None:
                        continue
                    rows.append(c)
                    if max_per_day and len(rows) - 1 >= max_per_day:
                        break
        with open(os.path.join(out, f"gdelt_{d}.csv"), "w", encoding="utf-8") as w:
            w.write("\n".join(rows) + "\n")
        man.write(f"{d}\t{url}\t{sha}\t{len(rows) - 1}\n")
        man.flush()
        print(d, len(rows) - 1, sha[:12], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
