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

.PHONY: all debug clean perft bench baseline doc

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

# API documentation (Doxygen; README.md is the main page). Needs doxygen + graphviz.
doc:
	doxygen Doxyfile
	@echo "docs -> doc/html/index.html"
