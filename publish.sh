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