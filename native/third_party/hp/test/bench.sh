#!/usr/bin/env bash
# Measure peak RSS and wall time for hp compress/decompress.
set -euo pipefail
HP=${HP:-build/hp}
FILE=${1:-data/proxy256k.xml}
MEM=${2:-22}
PEEKRSS=${PEEKRSS:-/tmp/peekrss}

if [ ! -x "$PEEKRSS" ]; then
  cat > /tmp/peekrss.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <string.h>
static long read_hwm(pid_t pid) {
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/status", (int)pid);
    FILE* f = fopen(path, "r");
    if (!f) return 0;
    char line[256];
    long hwm = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) sscanf(line + 6, "%ld", &hwm);
    }
    fclose(f);
    return hwm;
}
int main(int argc, char** argv) {
    if (argc < 2) return 1;
    pid_t pid = fork();
    if (pid < 0) return 1;
    if (pid == 0) { execvp(argv[1], &argv[1]); _exit(127); }
    long peak = 0;
    int status = 0;
    while (waitpid(pid, &status, WNOHANG) == 0) {
        long h = read_hwm(pid);
        if (h > peak) peak = h;
        usleep(5000);
    }
    long h = read_hwm(pid);
    if (h > peak) peak = h;
    printf("%ld\n", peak);
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
EOF
  gcc -O2 -o "$PEEKRSS" /tmp/peekrss.c
fi

ARC=/tmp/hp_bench.arc
OUT=/tmp/hp_bench.out
START=$(date +%s%3N)
PEAK_C=$("$PEEKRSS" "$HP" c --mem "$MEM" "$FILE" "$ARC" 2>/dev/null | tr -cd '0-9')
MID=$(date +%s%3N)
PEAK_D=$("$PEEKRSS" "$HP" d "$ARC" "$OUT" 2>/dev/null | tr -cd '0-9')
END=$(date +%s%3N)
IN=$(stat -c%s "$FILE")
ARC_SZ=$(stat -c%s "$ARC")
PEAK=$PEAK_C
if [ -n "$PEAK_D" ] && [ "$PEAK_D" -gt "$PEAK" ]; then PEAK=$PEAK_D; fi
sha_in=$(sha256sum "$FILE" | awk '{print $1}')
sha_out=$(sha256sum "$OUT" | awk '{print $1}')
if [ "$sha_in" != "$sha_out" ]; then
  echo "FAIL roundtrip sha mismatch"
  exit 1
fi
echo "input_bytes=$IN archive_bytes=$ARC_SZ peak_rss_kb=$PEAK compress_ms=$((MID-START)) decompress_ms=$((END-MID))"
