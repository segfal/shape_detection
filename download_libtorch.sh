#!/bin/bash

set -e

LIBTORCH_DIR="$HOME/libs/libtorch"
TORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.7.1.zip"

echo "🔽 Downloading LibTorch for macOS (CPU-only, arm64)..."
mkdir -p "$(dirname "$LIBTORCH_DIR")"
curl -L "$TORCH_URL" -o /tmp/libtorch-macos.zip

echo "📦 Extracting to $LIBTORCH_DIR ..."
rm -rf "$LIBTORCH_DIR"
unzip -q /tmp/libtorch-macos.zip -d "$(dirname "$LIBTORCH_DIR")"

# The zip extracts to libtorch/, so ensure the path is correct
if [ ! -d "$LIBTORCH_DIR" ]; then
    mv "$(dirname "$LIBTORCH_DIR")/libtorch" "$LIBTORCH_DIR"
fi

rm /tmp/libtorch-macos.zip

echo "✅ LibTorch is now available at $LIBTORCH_DIR" 