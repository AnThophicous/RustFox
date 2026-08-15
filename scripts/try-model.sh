#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$PWD/models}"
FOX="${FOX:-$PWD/build/fox}"

usage() {
    cat <<'EOF'
try-model.sh — download a GGUF and run it through fox

usage:
  scripts/try-model.sh <url> [-- fox run options...]
  scripts/try-model.sh --list

Nothing here is a recommendation of a particular model. These are just
small, standard llama-architecture files that make good first targets,
because they hold the architecture constant while you find out whether
the engine works at all.

  smol    SmolLM2 135M instruct, Q8_0, about 145 MB
          the fastest way to find out something is broken

  tiny    TinyLlama 1.1B chat, Q4_K_M, about 670 MB
          a real model, grouped query attention, still small

Anything using a custom quantisation type, linear or hybrid attention,
sub-layer norms, attention biases or a vision tower will be refused or
will warn. That is the engine telling the truth, not the engine failing.
EOF
}

resolve() {
    case "$1" in
        smol)
            echo "https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF/resolve/main/smollm2-135m-instruct-q8_0.gguf"
            ;;
        tiny)
            echo "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
            ;;
        *)
            echo "$1"
            ;;
    esac
}

if [ $# -lt 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ] || [ "$1" = "--list" ]; then
    usage
    exit 0
fi

URL="$(resolve "$1")"
shift

if [ ! -x "$FOX" ]; then
    echo "fox is not built yet. Run:" >&2
    echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j" >&2
    exit 1
fi

mkdir -p "$MODELS_DIR"
NAME="$(basename "${URL%%\?*}")"
DEST="$MODELS_DIR/$NAME"

if [ ! -f "$DEST" ]; then
    echo "downloading $NAME"
    curl -fL --progress-bar -o "$DEST.part" "$URL"
    mv "$DEST.part" "$DEST"
else
    echo "already have $NAME"
fi

ls -lh "$DEST"
echo

if [ "${1:-}" = "--" ]; then shift; fi

if [ $# -eq 0 ]; then
    set -- -p "The capital of France is" -n 32 --temp 0
fi

echo "\$ fox run $DEST $*"
echo
exec "$FOX" run "$DEST" "$@"
