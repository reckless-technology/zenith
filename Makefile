# Zenith build (C17). Single-shot whole-program compile so LTO sees everything (the engine is small).
# A shipped build would fan out per microarch (x86-64-v2/v3/v4); -march=native is for local dev.
# -D_POSIX_C_SOURCE is needed for clock_gettime/strtok_r under strict -std=c17 (not gnu17).
CC        = clang
STD       = -std=c17 -D_POSIX_C_SOURCE=200809L
OPT       = -O3 -march=native -funroll-loops -flto -DNDEBUG
WARN      = -Wall -Wextra -Wpedantic -Werror -Wshadow -Wno-unused-parameter
# Build number = git commit count (globally reproducible per commit; 0 outside a git checkout).
BUILD_NUM = $(shell git rev-list --count HEAD 2>/dev/null || echo 0)
VERSION   = -DZENITH_BUILD_NUMBER=$(BUILD_NUM)
CFLAGS    = $(STD) $(OPT) $(WARN) $(VERSION)
LDLIBS    = -lm -pthread
SRCS      = $(wildcard src/*.c)
HDRS      = $(wildcard src/*.h)
BIN       = zenith

.PHONY: all debug clean perft bench baseline doc check format hooks

all: $(BIN)

$(BIN): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(SRCS) -o $(BIN) $(LDLIBS)

# Correctness build: sanitizers on, optimizer light. Used to shake out movegen UB before trusting perft.
debug: $(SRCS) $(HDRS)
	$(CC) $(STD) $(VERSION) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(WARN) $(SRCS) -o $(BIN)-debug $(LDLIBS)

perft: $(BIN)
	./$(BIN) perft

bench: $(BIN)
	./$(BIN) bench

clean:
	rm -f $(BIN) $(BIN)-debug
	rm -rf doc/html

baseline: $(BIN)
	cp $(BIN) $(BIN)-base
	@echo "baseline -> $(BIN)-base"

# Full local test suite — every self-check gate (mirrors CI). Any failure aborts with a non-zero exit.
# nnuecheck runs only if the shipped net is present (nets/ is gitignored).
NET = nets/zenith-kb3.nnue
check: $(BIN)
	@echo "== perft ==";      ./$(BIN) perft | tail -1
	@./$(BIN) perft >/dev/null 2>&1 || { echo "perft FAILED"; exit 1; }
	@echo "== bench signature =="; \
	  sig=`./$(BIN) bench 13 | tail -1 | grep -oE '^[0-9]+'`; \
	  if [ "$$sig" = "3325894" ]; then echo "  $$sig PASS"; else echo "  $$sig FAIL (want 3325894)"; exit 1; fi
	@echo "== legalcheck ==";  ./$(BIN) legalcheck
	@echo "== seecheck ==";    ./$(BIN) seecheck
	@echo "== fuzzcheck ==";   ./$(BIN) fuzzcheck
	@echo "== bookcheck ==";   ./$(BIN) bookcheck
	@echo "== nnuecheck ==";   if [ -f $(NET) ]; then ./$(BIN) nnuecheck $(NET); else echo "  SKIP (no $(NET))"; fi
	@echo "make check: all gates passed"

# Format all C sources in place with the repo .clang-format.
format:
	clang-format -i $(SRCS) $(HDRS)

# Enable the versioned git hooks (clang-format pre-commit check). Run once per clone.
hooks:
	git config core.hooksPath .githooks
	@echo "git hooks enabled (core.hooksPath -> .githooks)"

# API documentation (Doxygen; README.md is the main page). Needs doxygen + graphviz.
doc:
	doxygen Doxyfile
	@echo "docs -> doc/html/index.html"
