#!/bin/bash
cd /mnt/c/Users/odinl/OneDrive/Desktop/Cypha || exit 1
exec python3 -u scripts/fetch_corpus.py --root /mnt/d/cypha_llm/corpus
