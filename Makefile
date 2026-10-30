# Zenith build (C17). Single-shot whole-program compile so LTO sees everything (the engine is small).
# A shipped build would fan out per microarch (x86-64-v2/v3/v4); -march=native is for local dev.
# -D_POSIX_C_SOURCE is needed for clock_gettime/strtok_r under strict -std=c17 (not gnu17).
# -D_DARWIN_C_SOURCE undoes that macro's visibility RESTRICTION on Apple headers (sys/sysctl.h needs the
# BSD types u_int/u_char that strict POSIX hides); it is inert on every other platform.
CC        = clang
STD       = -std=c17 -D_POSIX_C_SOURCE=200809L -D_DARWIN_C_SOURCE
# Target microarch. Default 'native' for local dev; a release fans out per microarch (e.g. ARCH=x86-64-v2
# for broad compatibility, ARCH=x86-64-v3 which guarantees BMI2 for `make pext`). See .github/workflows/release.yml.
# ARCH= (empty) omits -march entirely — a portable baseline, and what macos-check.yml uses. Apple clang used to
# reject -march=native on arm64; Apple clang 21 (Xcode 26) accepts it and resolves it to the host CPU
# (apple-m1 on an M1 Pro), so the default target builds on Apple silicon too. Keep ARCH= for portability.
ARCH      ?= native
ARCH_FLAG  = $(if $(ARCH),-march=$(ARCH))
OPT       = -O3 $(ARCH_FLAG) -funroll-loops -flto -DNDEBUG
WARN      = -Wall -Wextra -Wpedantic -Werror -Wshadow -Wno-unused-parameter
# The version, MAJOR.MINOR.BUILD, from git alone (tools/version.sh): the nearest annotated vMAJOR.MINOR tag and the
# commits since it; 0.0.0 with no such tag reachable. Compiled in as ZENITH_VERSION_STRING (src/version.h); nothing in
# the tree is edited to release. Evaluated once per make run (:=), so its warning shows once.
ZENITH_VERSION := $(shell tools/version.sh)
VERSION   = -DZENITH_VERSION_STRING='"$(ZENITH_VERSION)"'
# Rewritten only when the version changes, and a prerequisite of every binary, so a new commit or tag rebuilds them.
VERSION_STAMP = build/version.txt
CFLAGS    = $(STD) $(OPT) $(WARN) $(VERSION)
LDLIBS    = -lm -pthread
# The engines a release ships link their toolchain's runtime statically on Windows: MSYS2 CLANG64's -pthread is
# libwinpthread-1.dll, which a Windows without MSYS2 lacks, and an engine importing it does not start there
# (0xC0000135). -static links winpthread (and compiler-rt's builtins) into the binary; the C runtime stays the
# UCRT's api-ms-win-crt-*.dll sets, which are Windows' own from Windows 10 on. tools/windows-dependents.ps1 holds the
# shipped engines to that. Elsewhere, and in the debug, coverage and datagen builds, nothing changes.
RELEASE_LDFLAGS = $(if $(filter Windows_NT,$(OS)),-static)
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
NET       = nets/reckless-lc0-runf-sb870.nnue
EMBED_OBJ = $(BUILD_DIR)/embedded_net.o
# The shipped opening book (committed), embedded the same way: `setoption name OwnBook value true` works
# with no BookFile (OwnBook still defaults false — testing stays bookless).
BOOK_BIN  = books/zenith-book-r5.bin
EMBED_BOOK_OBJ = $(BUILD_DIR)/embedded_book.o

.PHONY: all debug clean perft bench baseline doc check coverage format hooks get-book pext tables datagen datagen-debug FORCE

all: $(BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(VERSION_STAMP): FORCE | $(BUILD_DIR)
	@[ "$$(cat $@ 2>/dev/null)" = "$(ZENITH_VERSION)" ] || echo "$(ZENITH_VERSION)" > $@

# The compiler and the target and optimisation flags, rewritten only when they change, and a prerequisite of the
# engine: without it `make ARCH=x86-64-v2` after `make` would find build/zenith up to date and leave the native build
# in place, so a check or a benchmark of the portable build would run the wrong binary.
FLAGS_STAMP = $(BUILD_DIR)/flags.txt
$(FLAGS_STAMP): FORCE | $(BUILD_DIR)
	@[ "$$(cat $@ 2>/dev/null)" = "$(CC) $(OPT)" ] || echo "$(CC) $(OPT)" > $@

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

$(BIN): $(SRCS) $(HDRS) $(INCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) $(VERSION_STAMP) $(FLAGS_STAMP) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN) $(LDLIBS) $(RELEASE_LDFLAGS)

# Correctness build: sanitizers on, optimizer light. Used to shake out movegen UB before trusting perft.
# -fno-sanitize-recover=undefined: UBSan reports undefined behaviour and then aborts, so a finding fails the run (and
# CI's sanitizers job); by default it prints and carries on, and the process still exits 0.
DEBUG_FLAGS = $(STD) $(VERSION) -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer $(WARN)
debug: $(SRCS) $(HDRS) $(INCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) $(VERSION_STAMP) | $(BUILD_DIR)
	$(CC) $(DEBUG_FLAGS) $(SRCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-debug $(LDLIBS)

# PEXT (BMI2) sliding-attack lookups instead of magic bitboards -> ./build/zenith-pext. Output is bit-identical
# to the magic build (same bench signature); ~2% faster perft / ~1.7% faster search on Intel Haswell+ and
# AMD Zen3+, but MUCH slower on AMD Zen1/Zen2 (microcoded pext) — so it is opt-in and magic stays the
# portable default. Needs a BMI2 target (the default -march=native provides it; a real release enables it
# only for the x86-64-v3+ microarch variants).
pext: $(SRCS) $(HDRS) $(INCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) $(VERSION_STAMP) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -DZENITH_USE_PEXT $(SRCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-pext $(LDLIBS) $(RELEASE_LDFLAGS)
	@echo "built $(BIN)-pext (PEXT/BMI2 sliding attacks; bit-identical to $(BIN))"

# NNUE training-data tools -> ./build/zenith-datagen + ./build/zenith-bullet2text. Standalone executables
# (they replaced the old `datagen`/`bullet2text` engine subcommands), same single-shot whole-program compile
# as the engine: the self-play generator runs real searches, so both link the engine core (LTO drops what
# the converter never calls). -Isrc lets datagen/ include the engine headers by name.
datagen: $(ENGINE_CORE_SRCS) $(HDRS) $(INCS) $(DATAGEN_SRCS) $(DATAGEN_HDRS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) $(VERSION_STAMP) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -Isrc $(ENGINE_CORE_SRCS) datagen/datagen.c datagen/datagen_main.c $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-datagen $(LDLIBS)
	$(CC) $(CFLAGS) -Isrc $(ENGINE_CORE_SRCS) datagen/datagen.c datagen/bullet2text_main.c $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(BIN)-bullet2text $(LDLIBS)
	@echo "built $(BIN)-datagen (self-play generator) and $(BIN)-bullet2text (bulletformat -> text)"

# Correctness builds of the datagen tools (ASan+UBSan, -O1). The tool sources live outside src/, so
# `make debug` no longer compiles them — this keeps them under the same sanitizers (CI builds this and
# smoke-runs the generator).
datagen-debug: $(ENGINE_CORE_SRCS) $(HDRS) $(INCS) $(DATAGEN_SRCS) $(DATAGEN_HDRS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) $(VERSION_STAMP) | $(BUILD_DIR)
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

# Every self-check gate but nnuecheck, run against the binary named by $(1), with tools/uci_check.py's extra flags in
# $(2): `make check` runs them on the release build, `make coverage` on the instrumented one.
define RUN_GATES
	@echo "== perft ==";      ./$(1) perft || { echo "perft FAILED"; exit 1; }
	@echo "== bench ==";       ./$(1) bench || { echo "bench FAILED"; exit 1; }
	@echo "== legalcheck ==";  ./$(1) legalcheck
	@echo "== seecheck ==";    ./$(1) seecheck
	@echo "== fuzzcheck ==";   ./$(1) fuzzcheck
	@echo "== bookcheck ==";   ./$(1) bookcheck
	@echo "== embedded book =="; ./$(1) embookcheck
	@#  ...plus the end-to-end UCI path the gate above cannot reach: OwnBook wiring -> bestmove from the book. A clock,
	@#  since the book is never probed for a fixed-depth, node-limited or infinite search.
	@printf 'setoption name OwnBook value true\nposition startpos\ngo wtime 60000 btime 60000\nquit\n' \
	  | ./$(1) | grep -q '^info string book move' \
	  && echo "[PASS] embook uci                             bestmove from book                              (setoption OwnBook true -> go)" \
	  || { echo "[FAIL] embook uci                             no book bestmove                                (setoption OwnBook true -> go)"; exit 1; }
	@echo "== uci ==";         python3 tools/uci_check.py $(2) ./$(1)
endef
NNUE_GATE = if [ -f $(NET) ]; then ./$(1) nnuecheck $(NET); else echo "  SKIP (no $(NET))"; fi

# Full local test suite — every self-check gate (mirrors CI). Any failure aborts with a non-zero exit.
check: $(BIN)
	$(call RUN_GATES,$(BIN))
	@echo "== nnuecheck ==";   $(call NNUE_GATE,$(BIN))
	@echo "make check: all gates passed"

# Line coverage of src/ by the gates, with LLVM's source-based coverage: an instrumented engine (-O2 -g, the same
# sources and embedded net and book, no LTO) in build/cov/, every `make check` gate run against it - the front-end
# gate with --terminal, which on Linux and macOS also runs legalcheck on a pseudo-terminal for its progress line - plus
# CI's threaded UCI smoke, each process writing build/cov/<pid>.profraw, then merged into build/cov/zenith.profdata and
# reported over src/: the table on stdout, build/cov/lcov.info for an editor or a coverage service, and an HTML tree
# under build/cov/html/. It fails when the lines covered fall under COVERAGE_MIN per cent of src/'s (CI's coverage job
# runs it on Linux). src/ is the engine every release ships; the developer tools no release ships - datagen/, trainer/,
# tools/ - are not measured. nnuecheck is the slowest gate instrumented (6.6 million nodes, each checked against a full
# refresh: some 30 s on an i9-13900HX), and the only one into most of nnue.c's incremental updates and the finny cache.
# The bench gate still asserts its node signature: the instrumentation counts, it changes nothing the search computes.
# Needs clang and its profile runtime (compiler-rt) and llvm-profdata / llvm-cov: on macOS Xcode's, through xcrun; on
# Linux whichever llvm-profdata is on the PATH, else the one versioned like clang (llvm-profdata-18 for clang 18) -
# LLVM_SUFFIX=-18, or LLVM_PROFDATA= and LLVM_COV=, name them outright; on Windows MSYS2 CLANG64's llvm-tools.
COV_DIR    = $(BUILD_DIR)/cov
COV_BIN    = $(COV_DIR)/zenith
COV_FLAGS  = $(STD) -O2 -g $(ARCH_FLAG) -DNDEBUG -fprofile-instr-generate -fcoverage-mapping $(WARN) $(VERSION)
COV_DATA   = $(COV_DIR)/zenith.profdata
COVERAGE_MIN ?= 90
# llvm-cov is a native program on Windows: it needs the .exe the linker added, where MSYS2's make and bash find
# build/cov/zenith without it.
EXE        = $(if $(filter Windows_NT,$(OS)),.exe)
ifeq ($(shell uname -s),Darwin)
LLVM_PROFDATA ?= xcrun llvm-profdata
LLVM_COV      ?= xcrun llvm-cov
else
LLVM_SUFFIX   ?= $(if $(shell command -v llvm-profdata 2>/dev/null),,-$(firstword $(subst ., ,$(shell $(CC) -dumpversion))))
LLVM_PROFDATA ?= llvm-profdata$(LLVM_SUFFIX)
LLVM_COV      ?= llvm-cov$(LLVM_SUFFIX)
endif

$(COV_DIR): | $(BUILD_DIR)
	mkdir -p $(COV_DIR)

$(COV_BIN): $(SRCS) $(HDRS) $(INCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) $(VERSION_STAMP) $(FLAGS_STAMP) | $(COV_DIR)
	$(CC) $(COV_FLAGS) $(SRCS) $(EMBED_OBJ) $(EMBED_BOOK_OBJ) -o $(COV_BIN) $(LDLIBS)

# %m: every process merges its counts into the one profile under a file lock (the runtime's online merging), so no
# run overwrites another's. A file per process (%p, the pid) is not enough: Windows reuses pids within a run, and the
# front-end gate starts some hundred processes, so a later one could replace the bench gate's profile.
coverage: export LLVM_PROFILE_FILE = $(COV_DIR)/zenith-%m.profraw
coverage: $(COV_BIN)
	@$(LLVM_PROFDATA) --version >/dev/null 2>&1 || { echo "coverage: $(LLVM_PROFDATA) not found (set LLVM_SUFFIX, or LLVM_PROFDATA and LLVM_COV)" >&2; exit 1; }
	@rm -f $(COV_DIR)/*.profraw $(COV_DATA)
	$(call RUN_GATES,$(COV_BIN),--terminal)
	@echo "== uci smoke ==";  printf 'uci\nisready\nsetoption name Threads value 2\nposition startpos moves e2e4 e7e5\ngo depth 10\nucinewgame\nposition startpos\ngo movetime 200\nquit\n' \
	  | ./$(COV_BIN) | grep -c '^bestmove ' | grep -qx 2 && echo "[PASS] uci smoke: two bestmoves" || { echo "[FAIL] uci smoke"; exit 1; }
	@echo "== nnuecheck =="; $(call NNUE_GATE,$(COV_BIN))
	$(LLVM_PROFDATA) merge -sparse $(COV_DIR)/*.profraw -o $(COV_DATA)
	@rm -f $(COV_DIR)/*.profraw
	$(LLVM_COV) export -format=lcov $(COV_BIN)$(EXE) -instr-profile=$(COV_DATA) $(SRCS) $(HDRS) > $(COV_DIR)/lcov.info
	$(LLVM_COV) show -format=html -output-dir=$(COV_DIR)/html $(COV_BIN)$(EXE) -instr-profile=$(COV_DATA) $(SRCS) $(HDRS)
	$(LLVM_COV) report $(COV_BIN)$(EXE) -instr-profile=$(COV_DATA) $(SRCS) $(HDRS) | tee $(COV_DIR)/report.txt
	@echo "coverage: build/cov/lcov.info, build/cov/html/index.html"
	@# The TOTAL row's line count and missed lines are its 8th and 9th fields.
	@awk -v min=$(COVERAGE_MIN) '$$1 == "TOTAL" { found = 1; covered = 100 * ($$8 - $$9) / $$8; \
	  printf "coverage: %.2f %% of src/ lines (minimum %s %%)\n", covered, min; exit !(covered >= min) } \
	  END { if (!found) { print "coverage: no TOTAL row in the report" > "/dev/stderr"; exit 1 } }' $(COV_DIR)/report.txt \
	  || { echo "coverage: FAILED - under COVERAGE_MIN" >&2; exit 1; }

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
