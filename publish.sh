#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./publish.sh <package_name>"
    echo "Example: ./publish.sh apply"
    exit 1
fi

PACKAGE_DIR="$1"

if [ ! -d "$PACKAGE_DIR" ]; then
    echo "Error: Directory '$PACKAGE_DIR' does not exist"
    exit 1
fi

echo "Publishing package: $PACKAGE_DIR"
echo ""

cd "$PACKAGE_DIR"

# Check if common is a symlink and handle it
COMMON_DIR="notbadai_${PACKAGE_DIR}/common"
SYMLINK_TARGET=""

if [ -L "$COMMON_DIR" ]; then
    echo "Detected symlink for common directory. Creating temporary copy..."

    # Save the symlink target before removing it
    SYMLINK_TARGET=$(readlink "$COMMON_DIR")
    echo "Symlink points to: $SYMLINK_TARGET"

    # Remove the symlink and copy the actual content
    rm "$COMMON_DIR"
    cp -r "$SYMLINK_TARGET" "$COMMON_DIR"
    echo "✓ Temporary copy created"
fi

# Cleanup function to restore symlink
cleanup() {
    if [ -n "$SYMLINK_TARGET" ]; then
        echo "Restoring common symlink..."
        rm -rf "$COMMON_DIR"
        ln -s "$SYMLINK_TARGET" "$COMMON_DIR"
        echo "✓ Symlink restored to $SYMLINK_TARGET"
    fi
}

# Set trap to ensure cleanup happens even if script fails
trap cleanup EXIT

echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info

echo "Building package..."
python -m build

echo "Build complete! Files in dist/:"
ls -lh dist/

echo ""
echo "Uploading to PyPI..."
twine upload dist/*

echo ""
echo "✓ Successfully built and uploaded $PACKAGE_DIR to PyPI!"