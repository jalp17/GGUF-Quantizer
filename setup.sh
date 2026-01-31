#!/bin/bash

# GGUF-Quantizer Setup Script for Linux/macOS
# This script clones llama.cpp and applies custom image quantization patches.

REPO_URL="https://github.com/ggerganov/llama.cpp.git"
TARGET_DIR="llama.cpp"

echo "=== GGUF-Quantizer Setup (Linux/macOS) ==="

if [ ! -d "$TARGET_DIR" ]; then
    echo "Cloning llama.cpp..."
    git clone --depth 1 $REPO_URL $TARGET_DIR
else
    echo "Directory $TARGET_DIR already exists. Skipping clone."
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

echo ""
echo "=== Success! ==="
echo "Now you can build llama.cpp using your preferred method (cmake or make)."
echo "Example: cd llama.cpp && cmake -B build && cmake --build build --config Release"
