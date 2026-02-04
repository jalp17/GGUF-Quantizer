#!/bin/bash

# GGUF-Quantizer Setup Script for Linux/macOS
# This script clones llama.cpp and applies custom image quantization patches.

REPO_URL="https://github.com/ggerganov/llama.cpp.git"
TARGET_DIR="llama.cpp"
# LTS Base Commit (v1.2.1 LTS - Master Sync 2026-02-01)
LTS_HASH="2634ed207a17db1a54bd8df0555bd8499a6ab691"
CLEAR_CACHE=false

# Parse arguments
for arg in "$@"; do
    if [ "$arg" == "--clear-cache" ]; then
        CLEAR_CACHE=true
    fi
done

echo -e "\033[0;36m=== GGUF-Quantizer Setup (Linux/macOS) ===\033[0m"

if [ "$CLEAR_CACHE" = true ]; then
    echo -e "\033[0;33mCleaning ccache...\033[0m"
    ccache -C
fi

if [ ! -d "$TARGET_DIR" ]; then
    echo "Cloning llama.cpp (LTS Version)..."
    git clone "$REPO_URL" "$TARGET_DIR"
    cd "$TARGET_DIR" || exit
    git checkout "$LTS_HASH"
    cd ..
else
    echo "Directory $TARGET_DIR already exists. Ensuring it's on LTS version..."
    cd "$TARGET_DIR" || exit
    git checkout "$LTS_HASH"
    cd ..
fi

cd $TARGET_DIR || exit

echo "Applying patches..."
for patch in ../patches/*.patch; do
    echo "Applying $patch..."
    git apply --verbose "$patch"
    if [ $? -ne 0 ]; then
        echo "Error applying $patch. Please check for conflicts."
        exit 1
    fi
done

done

echo ""
echo "=== Success! ==="
echo "Now you can build llama.cpp using your preferred method (cmake or make)."
echo "Example: cd llama.cpp && cmake -B build && cmake --build build --config Release"
