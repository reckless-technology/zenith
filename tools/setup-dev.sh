#!/usr/bin/env bash
# Set up a Linux or macOS machine to build, test, debug and format Zenith: the toolchain, clang-format and the editor,
# and with --training the NNUE trainer's Python environment. Each component is checked first and installed only when it
# is missing, so the script is safe to run again at any time. uv and clang-format go into the user's own profile
# (~/.local/bin); the system packages come from apt (Debian, Ubuntu, Mint: sudo asks for a password), and on macOS from
# Xcode's command-line tools alone. tools/setup-dev.ps1 is the same for Windows. VS Code's "setup environment" and
# "check environment" tasks run the one for the platform.
#
# What it sets up:
#   - clang (with its sanitizer runtimes, for make debug), make, python3 (the Makefile's table and embedding scripts),
#     git, zip and the debugger: apt's clang make python3 git zip gdb curl; on macOS Xcode's command-line tools, which
#     carry all of them and lldb in gdb's place
#   - uv, from Astral's release archive (its SHA-256 the one published beside it), into ~/.local/bin
#   - clang-format 22.1.8 through uv, the one version the pre-commit hook and CI's format gate accept
#   - the repo's git hooks (core.hooksPath .githooks) and the VS Code extensions .vscode/extensions.json recommends
#   - with --training: a .venv in the repo with torch and numpy for trainer/train.py - CUDA 13.0 wheels on Linux, the
#     default wheels (Metal) on macOS
#
#   tools/setup-dev.sh                # set up
#   tools/setup-dev.sh --check        # report only; exit 1 if anything is missing
#   tools/setup-dev.sh --no-persist   # add nothing to the shell profile's PATH (a test run in a scratch HOME)
#   tools/setup-dev.sh --training     # and the trainer's .venv (with --check: check it)
set -euo pipefail
cd "$(dirname "$0")/.."

check=0
persist=1
training=0
for arg in "$@"; do
    case "$arg" in
        --check) check=1 ;;
        --no-persist) persist=0 ;;
        --training) training=1 ;;
        -h | --help) sed -n '2,23p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

clang_format_version=22.1.8 # CI's format gate (.github/workflows/ci.yml) and the pre-commit hook
trainer_python=3.13
torch_index=https://download.pytorch.org/whl/cu130

# ---------------------------------------------------------------------------------------------------------------------
# Helpers (the same in every reckless-technology repo's setup script).
missing=0
ok() { printf '  \033[32mok\033[0m       %-24s %s\n' "$1" "$2"; }
miss() { missing=$((missing + 1)); printf '  \033[33mmissing\033[0m  %-24s %s\n' "$1" "$2"; }
installing() { printf '  \033[36minstall\033[0m  %-24s %s\n' "$1" "$2"; }
caution() { printf '  \033[35mwarning\033[0m  %-24s %s\n' "$1" "$2"; }
have() { command -v "$1" >/dev/null 2>&1; }
os=$(uname -s)

# Put a directory on this shell's PATH and, unless --no-persist, on every new shell's (the profiles a login shell,
# bash and zsh read).
add_to_path() {
    case ":$PATH:" in *":$1:"*) ;; *) PATH="$1:$PATH"; export PATH ;; esac
    [ "$persist" = 1 ] || return 0
    local line="export PATH=\"$1:\$PATH\""
    local profiles=("$HOME/.profile" "$HOME/.bashrc")
    [ "$os" = Darwin ] && profiles+=("$HOME/.zprofile")
    for f in "${profiles[@]}"; do
        grep -qxF "$line" "$f" 2>/dev/null || printf '\n%s\n' "$line" >>"$f"
    done
}

sha256_of() { if have sha256sum; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# Download a file and refuse it unless its SHA-256 is the one given.
fetch_verified() {
    curl -fsSL "$1" -o "$3"
    local got
    got=$(sha256_of "$3")
    if [ "$got" != "$2" ]; then rm -f "$3"; echo "SHA-256 mismatch for $1 (got $got, want $2)" >&2; exit 1; fi
}

sudo_cmd() { if [ "$(id -u)" = 0 ]; then "$@"; else sudo "$@"; fi; }

# apt packages that are not installed yet, installed together (Debian, Ubuntu, Mint).
apt_packages() {
    local need=()
    for p in "$@"; do
        dpkg-query -W -f='${Status}' "$p" 2>/dev/null | grep -q 'install ok installed' || need+=("$p")
    done
    if [ ${#need[@]} = 0 ]; then ok 'apt packages' "$*"; return; fi
    if [ "$check" = 1 ]; then miss 'apt packages' "${need[*]}"; return; fi
    installing 'apt packages' "${need[*]}"
    sudo_cmd apt-get update -q
    sudo_cmd env DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${need[@]}"
}

# macOS: Xcode's command-line tools, which carry clang, make, git, python3, zip and lldb.
macos_base() {
    if xcode-select -p >/dev/null 2>&1; then ok 'Xcode command-line tools' "$(xcode-select -p)"
    elif [ "$check" = 1 ]; then miss 'Xcode command-line tools' 'xcode-select --install'
    else
        installing 'Xcode command-line tools' 'a dialog opens: install, then run this script again'
        xcode-select --install || true
        exit 0
    fi
    local lacking=()
    for tool in clang make git python3 zip lldb curl; do have "$tool" || lacking+=("$tool"); done
    if [ ${#lacking[@]} = 0 ]; then ok 'toolchain' "$(clang --version | head -1), $(python3 --version)"
    else miss 'toolchain' "not on the PATH: ${lacking[*]}"; fi
}

linux_base() {
    if have apt-get; then apt_packages "$@"
    else
        echo "  This is not a Debian-family Linux: install the equivalents of these packages, then run again: $*" >&2
        if [ "$check" = 1 ]; then missing=$((missing + 1)); fi
    fi
}

set_git_hooks() {
    [ -d .githooks ] || return 0
    if [ "$(git config core.hooksPath || true)" = .githooks ]; then ok 'git hooks' 'core.hooksPath .githooks'
    elif [ "$check" = 1 ]; then miss 'git hooks' 'core.hooksPath is not .githooks'
    else git config core.hooksPath .githooks; installing 'git hooks' 'core.hooksPath .githooks'; fi
}

# The extensions .vscode/extensions.json recommends, when VS Code's command line is on the PATH.
install_vscode_extensions() {
    have code && [ -f .vscode/extensions.json ] || return 0
    local installed
    installed=$(code --list-extensions 2>/dev/null || true)
    for ext in $(sed -e 's://.*$::' .vscode/extensions.json | tr -d '\n' | sed -e 's/.*"recommendations"[^[]*\[//' -e 's/\].*//' | tr -d '" ' | tr ',' ' '); do
        if printf '%s\n' "$installed" | grep -qixF "$ext"; then ok 'VS Code extension' "$ext"
        elif [ "$check" = 1 ]; then miss 'VS Code extension' "$ext"
        else installing 'VS Code extension' "$ext"; code --install-extension "$ext" --force >/dev/null; fi
    done
}

# The version a clang-format reports ("clang-format version 22.1.8 (...)"), or nothing.
clang_format_version_of() { "$1" --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9]*\.[0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1; }

# ---------------------------------------------------------------------------------------------------------------------
echo "zenith: $([ "$check" = 1 ] && echo checking || echo 'setting up') the $os development environment"

case "$os" in
    Linux) linux_base clang make python3 git zip gdb curl ;;
    Darwin) macos_base ;;
    *) echo "unsupported system: $os (tools/setup-dev.ps1 is the Windows script)" >&2; exit 2 ;;
esac

# uv, which installs clang-format (and, with --training, the trainer's Python): Astral's installer's directory.
uv_bin="${XDG_BIN_HOME:-$HOME/.local/bin}"
[ -x "$uv_bin/uv" ] && add_to_path "$uv_bin"
if have uv; then ok 'uv' "$(uv --version)"
elif [ "$check" = 1 ]; then miss 'uv' "Astral's Python tool, which installs clang-format"
else
    case "$os-$(uname -m)" in
        Linux-x86_64) triple=x86_64-unknown-linux-gnu ;;
        Linux-aarch64) triple=aarch64-unknown-linux-gnu ;;
        Darwin-arm64) triple=aarch64-apple-darwin ;;
        Darwin-x86_64) triple=x86_64-apple-darwin ;;
        *) echo "no uv build for $os $(uname -m)" >&2; exit 1 ;;
    esac
    # Astral's release archive, as its installer fetches it, held to the SHA-256 published beside it.
    installing 'uv' "uv-$triple.tar.gz from Astral's latest release, to $uv_bin"
    scratch=$(mktemp -d)
    url="https://github.com/astral-sh/uv/releases/latest/download/uv-$triple.tar.gz"
    fetch_verified "$url" "$(curl -fsSL "$url.sha256" | cut -d' ' -f1)" "$scratch/uv.tar.gz"
    tar -xzf "$scratch/uv.tar.gz" -C "$scratch"
    mkdir -p "$uv_bin"
    cp "$scratch/uv-$triple/uv" "$scratch/uv-$triple/uvx" "$uv_bin/"
    rm -rf "$scratch"
    add_to_path "$uv_bin"
fi

# clang-format 22.1.8, the first on the PATH: what the pre-commit hook and make format find. A different version
# restyles code that is already clean, and the hook then skips its check. apt's clang-format-18 is not it, and on Linux
# ~/.local/bin must come before /usr/bin.
cf_version=""
have clang-format && cf_version=$(clang_format_version_of clang-format)
if [ "$cf_version" = "$clang_format_version" ]; then ok 'clang-format' "$cf_version ($(command -v clang-format))"
elif [ "$check" = 1 ]; then
    miss 'clang-format' "$clang_format_version (first on the PATH: ${cf_version:+$cf_version, }$(command -v clang-format || echo none))"
elif have uv; then
    installing 'clang-format' "$clang_format_version through uv tool install"
    uv tool install --force "clang-format==$clang_format_version"
    tool_bin=$(uv tool dir --bin)
    add_to_path "$tool_bin"
    hash -r
    cf_version=$(clang_format_version_of clang-format)
    if [ "$cf_version" != "$clang_format_version" ]; then
        caution 'clang-format' "$(command -v clang-format) ($cf_version) shadows uv's in $tool_bin: remove it, or put $tool_bin ahead of it"
    fi
fi

set_git_hooks
install_vscode_extensions

# The NNUE trainer's .venv: torch and numpy at trainer/requirements.txt's floors - on Linux from PyTorch's CUDA 13.0
# index (with PyPI for everything that index lacks), on macOS PyPI's own wheels, which run on Metal.
if [ "$training" = 1 ]; then
    venv_python=.venv/bin/python
    probe="import torch, numpy
c = torch.cuda.is_available()
m = torch.backends.mps.is_available()
print(torch.__version__, numpy.__version__, 'CUDA on ' + torch.cuda.get_device_name(0) if c else ('Metal' if m else ''), sep='|')"
    state=""
    [ -x "$venv_python" ] && state=$("$venv_python" -c "$probe" 2>/dev/null | tail -1 || true)
    if [ -z "$state" ] && [ "$check" = 0 ]; then
        have uv || { echo 'uv is needed for the trainer .venv' >&2; exit 1; }
        if [ ! -x "$venv_python" ]; then
            installing 'trainer .venv' "Python $trainer_python through uv"
            uv venv .venv --python "$trainer_python"
        fi
        if [ "$os" = Darwin ]; then
            installing 'trainer .venv' 'torch and numpy, from trainer/requirements.txt'
            uv pip install --python "$venv_python" -r trainer/requirements.txt
        else
            installing 'trainer .venv' 'torch (CUDA 13.0) and numpy, from trainer/requirements.txt'
            uv pip install --python "$venv_python" -r trainer/requirements.txt --index-url "$torch_index" \
                --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
        fi
        state=$("$venv_python" -c "$probe" 2>/dev/null | tail -1 || true)
    fi
    if [ -z "$state" ]; then miss 'trainer .venv' '.venv with torch and numpy (tools/setup-dev.sh --training)'
    else
        IFS='|' read -r torch_version numpy_version accelerator <<<"$state"
        if [ -n "$accelerator" ]; then ok 'trainer .venv' "torch $torch_version, numpy $numpy_version, $accelerator"
        else caution 'trainer .venv' "torch $torch_version, numpy $numpy_version, but no GPU: training runs on the CPU"; fi
    fi
fi

if [ "$check" = 1 ]; then
    if [ "$missing" -gt 0 ]; then echo "$missing missing: run tools/setup-dev.sh (VS Code: the 'setup environment' task)."; exit 1; fi
    echo 'Everything is in place.'
else
    echo 'Done. Open a new terminal (or restart VS Code) for PATH changes to reach it, then: make check'
fi
