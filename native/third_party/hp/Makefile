# hp — Cypha Hutter Prize core
#
#   make              release build (XSIMD=1 default)
#   make XSIMD=0      portable scalar build
#   make ASAN=1       AddressSanitizer build (-O1 -g)
#   make asan-test    ASAN build + proxy roundtrip smoke test
#   make gdb-test     gdb batch compress/decompress smoke test
#
CXX      ?= g++
XSIMD    ?= 1
ASAN     ?= 0
BIN      ?= build
XSIMD_FLAGS :=
ifeq ($(XSIMD),1)
  XSIMD_FLAGS := -DHP_XSIMD=1 -msse4.1 -Ithird_party/xsimd/include
else
  XSIMD_FLAGS := -DHP_XSIMD=0
endif
OPT ?= -O2
SAN_FLAGS :=
LDFLAGS ?=
ifeq ($(ASAN),1)
  OPT := -O1
  SAN_FLAGS := -fsanitize=address -fno-omit-frame-pointer -g
  LDFLAGS := $(SAN_FLAGS)
endif
CXXFLAGS ?= -std=c++17 -Wall $(OPT) -Iinclude $(XSIMD_FLAGS) $(SAN_FLAGS)

all: $(BIN)/hp $(BIN)/dump_experts $(BIN)/bench

$(BIN):
	mkdir -p $(BIN)

$(BIN)/hp: src/main.cpp $(wildcard include/hp/*.hpp) | $(BIN)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

$(BIN)/dump_experts: tools/dump_experts.cpp $(wildcard include/hp/*.hpp) | $(BIN)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

$(BIN)/bench: tools/bench.cpp $(wildcard include/hp/*.hpp) | $(BIN)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

test: all
	bash test/no_float.sh
	HP=$(BIN)/hp bash test/roundtrip.sh $(FILES)

determinism:
	bash test/determinism.sh $(FILE)

asan-test:
	$(MAKE) ASAN=1 XSIMD=$(XSIMD) clean all
	bash test/no_float.sh
	HP=$(BIN)/hp bash test/roundtrip.sh data/proxy64k.xml data/proxy256k.xml

gdb-test:
	bash test/gdb.sh $(FILE)

clean:
	rm -rf $(BIN)

.PHONY: all test determinism asan-test gdb-test clean
