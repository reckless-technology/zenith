# Zenith C17 port. Single-shot whole-program compile so LTO sees everything (mirrors the C++ build).
# -D_POSIX_C_SOURCE exposes clock_gettime/strtok_r/strdup under strict -std=c17.
CC      = clang
STD     = -std=c17 -D_POSIX_C_SOURCE=200809L
OPT     = -O3 -march=native -funroll-loops -flto -DNDEBUG
WARN    = -Wall -Wextra -Wpedantic -Werror -Wshadow -Wno-unused-parameter
CFLAGS  = $(STD) $(OPT) $(WARN)
LDLIBS  = -lm -pthread
SRCS    = $(wildcard src/*.c)
HDRS    = $(wildcard src/*.h)
BIN     = zenith-c

.PHONY: all debug clean perft bench

all: $(BIN)

$(BIN): $(SRCS) $(HDRS)
	$(CC) $(CFLAGS) $(SRCS) -o $(BIN) $(LDLIBS)

# Correctness build: sanitizers on, optimizer light.
debug: $(SRCS) $(HDRS)
	$(CC) $(STD) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(WARN) $(SRCS) -o $(BIN)-debug $(LDLIBS)

perft: $(BIN)
	./$(BIN) perft

bench: $(BIN)
	./$(BIN) bench

clean:
	rm -f $(BIN) $(BIN)-debug
