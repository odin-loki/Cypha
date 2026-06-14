#!/usr/bin/env bash
# Download WikiText-2 raw tokens into bench/data/wikitext2/wikitext-2/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="$REPO_ROOT/bench/data/wikitext2"
WIKITEXT_DIR="$DEST_DIR/wikitext-2"
URLS=(
  "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
  "https://raw.githubusercontent.com/LogSSim/deeplearning_d2l_classes/main/class14_BERT/wikitext-2-v1.zip"
)
ZIP_FILE="${TMPDIR:-/tmp}/wikitext-2-v1.zip"

valid_zip() {
  [[ -f "$1" ]] || return 1
  [[ "$(wc -c <"$1" | tr -d ' ')" -ge 1024 ]] || return 1
  local magic
  magic="$(head -c 2 "$1" || true)"
  [[ "$magic" == $'PK' ]]
}

required=(
  "$WIKITEXT_DIR/wiki.train.tokens"
  "$WIKITEXT_DIR/wiki.valid.tokens"
  "$WIKITEXT_DIR/wiki.test.tokens"
)
all_present=1
for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    all_present=0
    break
  fi
done
if [[ "$all_present" -eq 1 ]]; then
  echo "WikiText-2 already present at $WIKITEXT_DIR"
  exit 0
fi

mkdir -p "$DEST_DIR"
downloaded=0
for url in "${URLS[@]}"; do
  echo "Downloading WikiText-2 from $url ..."
  if command -v curl >/dev/null 2>&1; then
    if curl -fsSL "$url" -o "$ZIP_FILE" && valid_zip "$ZIP_FILE"; then
      downloaded=1
      break
    fi
  elif command -v wget >/dev/null 2>&1; then
    if wget -q -O "$ZIP_FILE" "$url" && valid_zip "$ZIP_FILE"; then
      downloaded=1
      break
    fi
  else
    echo "Need curl or wget to download WikiText-2" >&2
    exit 1
  fi
  rm -f "$ZIP_FILE"
done
if [[ "$downloaded" -ne 1 ]]; then
  echo "Could not download WikiText-2 from any configured URL" >&2
  exit 1
fi

extract_root="$(mktemp -d)"
cleanup() {
  rm -rf "$extract_root"
  rm -f "$ZIP_FILE"
}
trap cleanup EXIT

unzip -q "$ZIP_FILE" -d "$extract_root"
inner="$extract_root/wikitext-2"
if [[ ! -d "$inner" ]]; then
  echo "Expected wikitext-2/ inside archive" >&2
  exit 1
fi
rm -rf "$WIKITEXT_DIR"
mkdir -p "$WIKITEXT_DIR"
cp -a "$inner/." "$WIKITEXT_DIR/"

for path in "${required[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing after extract: $path" >&2
    exit 1
  fi
done
echo "WikiText-2 installed to $WIKITEXT_DIR"
