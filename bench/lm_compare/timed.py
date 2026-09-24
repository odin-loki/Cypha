#!/usr/bin/env python3
"""timed.py OUT.json CMD...: run CMD, record wall time, CPU time and peak RSS."""
import json
import resource
import subprocess
import sys
import time

t0 = time.perf_counter()
r = subprocess.run(sys.argv[2:], capture_output=True, text=True)
wall = time.perf_counter() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
json.dump({"cmd": sys.argv[2:], "wall_seconds": wall, "cpu_seconds": ru.ru_utime + ru.ru_stime,
           "peak_rss_mb": ru.ru_maxrss / 1024, "returncode": r.returncode, "stdout": r.stdout[-4000:]},
          open(sys.argv[1], "w"), indent=1)
sys.stdout.write(r.stdout)
sys.exit(r.returncode)
