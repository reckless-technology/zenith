# Zenith build (C17). Single-shot whole-program compile so LTO sees everything (the engine is small).
# A shipped build would fan out per microarch (x86-64-v2/v3/v4); -march=native is for local dev.
# -D_POSIX_C_SOURCE is needed for clock_gettime/strtok_r under strict -std=c17 (not gnu17).
# -D_DARWIN_C_SOURCE undoes that macro's visibility RESTRICTION on Apple headers (sys/sysctl.h needs the
# BSD types u_int/u_char that strict POSIX hides); it is inert on every other platform.
CC        = clang
STD       = -std=c17 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE
# Target microarch. Default 'native' for local dev; a release fans out per microarch (e.g. ARCH=x86-64-v2
# for broad compatibility, ARCH=x86-64-v3 which guarantees BMI2 for `make pext`). See .github/workflows/release.yml.
# ARCH= (empty) omits -march entirely — a portable baseline, needed on Apple-clang arm64 (no -march=native).
ARCH      ?= native
ARCH_FLAG  = $(if $(ARCH),-march=$(ARCH))
OPT       = -O3 $(ARCH_FLAG) -funroll-loops -flto -DNDEBUG
WARN      = -Wall -Wextra -Wpedantic -Werror -Wshadow -Wno-unused-parameter
# Build number = git commit count (globally reproducible per commit; 0 outside a git checkout).
BUILD_NUM = $(shell git rev-list --count HEAD 2>/dev/null || echo 0)
VERSION   = -DZENITH_BUILD_NUMBER=$(BUILD_NUM)
CFLAGS    = $(STD) $(OPT) $(WARN) $(VERSION)
LDLIBS    = -lm -pthread
SRCS      = $(wildcard src/*.c)
HDRS      = $(wildcard src/*.h)
# Generated constant tables (committed; rebuilt only by an explicit `make tables`). They are #included by
# .c files, so every binary must depend on them or edits leave a stale build.
INCS      = $(wildcard src/generated/*.inc)
# Training-data tooling (datagen/): built only by `make datagen`, never into the engine binary. The engine
# core (everything but src/main.c — the tools bring their own main()) is compiled into the tools so the
# self-play generator can run real searches.
DATAGEN_SRCS     = $(wildcard datagen/*.c)
DATAGEN_HDRS     = $(wildcard datagen/*.h)
ENGINE_CORE_SRCS = $(filter-out src/main.c,$(SRCS))
# All build outputs land in ./build (created on demand); `make clean` just removes it.
BUILD_DIR = build
BIN       = $(BUILD_DIR)/zenith
# The shipped net (committed). It is embedded into every binary at build time (below) and doubles as the
# `make check` nnuecheck target: the engine falls back to the embedded copy whenever EvalFile is unset or
# fails to load, so a bare binary is always full NNUE strength with no external files.
NET       = nets/zenith-cap1.nnue
EMBED_OBJ = $(BUILD_DIR)/embedded_net.o
# The shipped opening book (committed), embedded the same way: `setoption name OwnBook value true` works
# with no BookFile (OwnBook still defaults false — testing stays bookless).
BOOK_BIN  = books/zenith-book-r4.bin
EMBED_BOOK_OBJ = $(BUILD_DIR)/embedded_book.o

.PHONY: all debug clean perft bench baseline doc check format hooks get-book pext tables datagen datagen-debug

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# The embedded net: .nnue -> generated C array -> one cached object file. Regenerated/recompiled only
# when the net (or the generator) changes, so the 6.3MB array costs nothing on normal rebuilds. Plain
# data — no -march or LTO needed, and the same object links into the ASan builds.
$(BUILD_DIR)/embedded_net.c: $(NET) tools/embed_net.py | $(BUILD_DIR)
	python3 tools/embed_net.py $(NET) $@
$(EMBED_OBJ): $(BUILD_DIR)/embedded_net.c
	$(CC) $(STD) -O1 -c $< -o $@
$(BUILD_DIR)/embedded_book.c: $(BOOK_BIN) tools/embed_book.py | $(BUILD_DIR)
	python3 tools/embed_book.py $(BOOK_BIN) $@
$(EMBED_BOOK_OBJ): $(BUILD_DIR)/embedded_book.c
	$(CC) $(STD) -O1 -c $< -o $@

$(BIN): $(SRCS) $(HDRS) $(INCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN) $(LDLIBS)

# Correctness build: sanitizers on, optimizer light. Used to shake out movegen UB before trusting perft.
DEBUG_FLAGS = $(STD) $(VERSION) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(WARN)
debug: $(SRCS) $(HDRS) $(INCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) | $(BUILD_DIR)
	$(CC) $(DEBUG_FLAGS) $(SRCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-debug $(LDLIBS)

# PEXT (BMI2) sliding-attack lookups instead of magic bitboards -> ./build/zenith-pext. Output is bit-identical
# to the magic build (same bench signature); ~2% faster perft / ~1.7% faster search on Intel Haswell+ and
# AMD Zen3+, but MUCH slower on AMD Zen1/Zen2 (microcoded pext) — so it is opt-in and magic stays the
# portable default. Needs a BMI2 target (the default -march=native provides it; a real release enables it
# only for the x86-64-v3+ microarch variants).
pext: $(SRCS) $(HDRS) $(INCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DZENITH_USE_PEXT $(SRCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-pext $(LDLIBS)
	@echo "built $(BIN)-pext (PEXT/BMI2 sliding attacks; bit-identical to $(BIN))"

# NNUE training-data tools -> ./build/zenith-datagen + ./build/zenith-bullet2text. Standalone executables
# (they replaced the old `datagen`/`bullet2text` engine subcommands), same single-shot whole-program compile
# as the engine: the self-play generator runs real searches, so both link the engine core (LTO drops what
# the converter never calls). -Isrc lets datagen/ include the engine headers by name.
datagen: $(ENGINE_CORE_SRCS) $(HDRS) $(INCS) $(DATAGEN_SRCS) $(DATAGEN_HDRS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc $(ENGINE_CORE_SRCS) datagen/datagen.c datagen/datagen_main.c $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-datagen $(LDLIBS)
	$(CC) $(CFLAGS) -Isrc $(ENGINE_CORE_SRCS) datagen/datagen.c datagen/bullet2text_main.c $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-bullet2text $(LDLIBS)
	@echo "built $(BIN)-datagen (self-play generator) and $(BIN)-bullet2text (bulletformat -> text)"

# Correctness builds of the datagen tools (ASan+UBSan, -O1). The tool sources live outside src/, so
# `make debug` no longer compiles them — this keeps them under the same sanitizers (CI builds this and
# smoke-runs the generator).
datagen-debug: $(ENGINE_CORE_SRCS) $(HDRS) $(INCS) $(DATAGEN_SRCS) $(DATAGEN_HDRS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) | $(BUILD_DIR)
	$(CC) $(DEBUG_FLAGS) -Isrc $(ENGINE_CORE_SRCS) datagen/datagen.c datagen/datagen_main.c $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-datagen-debug $(LDLIBS)
	$(CC) $(DEBUG_FLAGS) -Isrc $(ENGINE_CORE_SRCS) datagen/datagen.c datagen/bullet2text_main.c $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-bullet2text-debug $(LDLIBS)
	@echo "built $(BIN)-datagen-debug and $(BIN)-bullet2text-debug (ASan+UBSan)"

perft: $(BIN)
	./$(BIN) perft

bench: $(BIN)
	./$(BIN) bench

clean:
	rm -rf $(BUILD_DIR) doc/html

baseline: $(BIN)
	cp $(BIN) $(BIN)-base
	@echo "baseline -> $(BIN)-base"

# Full local test suite — every self-check gate (mirrors CI). Any failure aborts with a non-zero exit.
check: $(BIN)
	@echo "== perft ==";      ./$(BIN) perft || { echo "perft FAILED"; exit 1; }
	@echo "== bench ==";       ./$(BIN) bench || { echo "bench FAILED"; exit 1; }
	@echo "== legalcheck ==";  ./$(BIN) legalcheck
	@echo "== seecheck ==";    ./$(BIN) seecheck
	@echo "== fuzzcheck ==";   ./$(BIN) fuzzcheck
	@echo "== bookcheck ==";   ./$(BIN) bookcheck
	@echo "== embedded book =="; printf 'setoption name OwnBook value true\nposition startpos\ngo depth 1\nquit\n' \
	  | ./$(BIN) | grep -qE 'bestmove (e2e4|d2d4|c2c4|g1f3)' \
	  && echo "  [PASS] embedded book probes from startpos" || { echo "embedded book FAILED"; exit 1; }
	@echo "== nnuecheck ==";   if [ -f $(NET) ]; then ./$(BIN) nnuecheck $(NET); else echo "  SKIP (no $(NET))"; fi
	@echo "make check: all gates passed"

# Regenerate the committed constant tables (Zobrist keys, PeSTO tables, Q28 ln table). Deliberately manual —
# these constants are part of the engine's identity (the bench signature depends on every value), so they
# change only on an explicit run of this target, never as a build side effect. Needs only a system
# python3 (the generator is stdlib-only — the training .venv is NOT required).
tables:
	python3 tools/generate_tables.py

# Format all C sources in place with the repo .clang-format.
format:
	clang-format -i $(SRCS) $(HDRS) $(DATAGEN_SRCS) $(DATAGEN_HDRS)

# Enable the versioned git hooks (clang-format pre-commit check). Run once per clone.
hooks:
	git config core.hooksPath .githooks
	@echo "git hooks enabled (core.hooksPath -> .githooks)"

# API documentation (Doxygen; README.md is the main page). Needs doxygen + graphviz.
doc:
	doxygen Doxyfile
	@echo "docs -> doc/html/index.html"

# Download a free Polyglot opening book (no book is committed; books/ is gitignored). This is
# performance.bin (~93k positions) as shipped in the GPL-3.0 python-chess repo, license-compatible with
# Zenith's own GPL-3.0-or-later; original source is the free WBEC-Ridderkerk collection. Idempotent: a
# correct existing copy is left untouched. Then: setoption name BookFile value $(BOOK) / OwnBook true.
BOOK     = books/performance.bin
BOOK_URL = https://raw.githubusercontent.com/niklasf/python-chess/master/data/polyglot/performance.bin
get-book:
	@if [ -f "$(BOOK)" ] && [ $$(( $$(wc -c < "$(BOOK)") % 16 )) -eq 0 ] && [ -s "$(BOOK)" ]; then \
	  echo "book already present: $(BOOK) ($$(wc -c < "$(BOOK)") bytes)"; \
	else \
	  mkdir -p "$(dir $(BOOK))"; \
	  echo "downloading $(BOOK) <- $(BOOK_URL)"; \
	  if command -v curl >/dev/null 2>&1; then curl -fSL --retry 3 -o "$(BOOK).tmp" "$(BOOK_URL)"; \
	  elif command -v wget >/dev/null 2>&1; then wget -O "$(BOOK).tmp" "$(BOOK_URL)"; \
	  else echo "need curl or wget to download the book" >&2; exit 1; fi; \
	  sz=$$(wc -c < "$(BOOK).tmp"); \
	  if [ "$$sz" -gt 0 ] && [ $$(( sz % 16 )) -eq 0 ]; then mv "$(BOOK).tmp" "$(BOOK)"; \
	  else rm -f "$(BOOK).tmp"; echo "downloaded file is not a valid Polyglot book ($$sz bytes, not a multiple of 16)" >&2; exit 1; fi; \
	  echo "book -> $(BOOK) ($$sz bytes, $$(( sz / 16 )) entries)"; \
	fi
	@echo 'enable it in UCI:  setoption name BookFile value $(BOOK)  /  setoption name OwnBook value true'
