#!/bin/bash

# 🐍 Poetry Installation Script for Shape Recognition
# This script installs Poetry and sets up the Python environment

set -e

echo "🐍 Installing Poetry for Shape Recognition"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Poetry is already installed
if command -v poetry &> /dev/null; then
    print_success "Poetry is already installed"
    poetry --version
else
    print_status "Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    print_success "Poetry installed successfully"
fi

# Check if we're in the project directory
if [ ! -f "pyproject.toml" ]; then
    print_warning "pyproject.toml not found. Please run this script from the project root directory."
    exit 1
fi

print_status "Installing Python dependencies..."
poetry install

print_success "Poetry environment setup completed!"
echo ""
echo "Next steps:"
echo "1. Activate the Poetry environment: poetry shell"
echo "2. Train the model: poetry run train-model"
echo "3. Test setup: poetry run test-setup"
echo "4. Build C++ application: ./setup.sh"
echo ""
echo "Or run commands directly:"
echo "  poetry run train-model"
echo "  poetry run test-setup" 