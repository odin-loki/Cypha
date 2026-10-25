#!/bin/bash
cd /mnt/c/Users/odinl/OneDrive/Desktop/Cypha || exit 1
g++ -std=c++23 -O2 -o /tmp/trie_stream_train native/tools/trie_stream_train.cpp || exit 1
exec taskset -c 0 /tmp/trie_stream_train /mnt/d/cypha_llm/corpus
