# Zenith build. Single-shot whole-program compile so LTO sees everything (the engine is small).
# A shipped build would fan out per microarch (x86-64-v2/v3/v4); -march=native is for local dev.
CXX       = clang++
STD       = -std=c++20
OPT       = -O3 -march=native -funroll-loops -flto -DNDEBUG
WARN      = -Wall -Wextra -Wshadow -Wno-unused-parameter
CXXFLAGS  = $(STD) $(OPT) $(WARN)
SRCS      = $(wildcard src/*.cpp)
HDRS      = $(wildcard src/*.h)
BIN       = zenith

.PHONY: all debug clean perft bench

all: $(BIN)

$(BIN): $(SRCS) $(HDRS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(BIN)

# Correctness build: sanitizers on, optimizer light. Used to shake out movegen UB before trusting perft.
debug: $(SRCS) $(HDRS)
	$(CXX) $(STD) -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer $(WARN) $(SRCS) -o $(BIN)-debug

perft: $(BIN)
	./$(BIN) perft

bench: $(BIN)
	./$(BIN) bench

clean:
	rm -f $(BIN) $(BIN)-debug
