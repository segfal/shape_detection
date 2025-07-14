#!/bin/bash

# 🔧 Shape Recognizer Setup Script for macOS
# This script automates the setup process for the C++ shape recognizer

set -e  # Exit on any error

echo "🔍 Shape Recognizer Setup for macOS"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Homebrew is installed
check_homebrew() {
    print_status "Checking Homebrew installation..."
    if ! command -v brew &> /dev/null; then
        print_warning "Homebrew not found. Installing..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
            eval "$(/opt/homebrew/bin/brew shellenv)"
        fi
    else
        print_success "Homebrew is already installed"
    fi
}

# Install OpenCV
install_opencv() {
    print_status "Installing OpenCV..."
    if brew list opencv &> /dev/null; then
        print_success "OpenCV is already installed"
    else
        brew install opencv
        print_success "OpenCV installed successfully"
    fi
}

# Check Poetry installation
check_poetry() {
    print_status "Checking Poetry installation..."
    if ! command -v poetry &> /dev/null; then
        print_warning "Poetry not found. Installing..."
        curl -sSL https://install.python-poetry.org | python3 -
        print_success "Poetry installed successfully"
        
        # Add Poetry to PATH
        export PATH="$HOME/.local/bin:$PATH"
    else
        print_success "Poetry is already installed"
    fi
}

# Install Python dependencies with Poetry
install_python_deps() {
    print_status "Installing Python dependencies with Poetry..."
    if [ -f "pyproject.toml" ]; then
        poetry install
        print_success "Python dependencies installed successfully"
    else
        print_error "pyproject.toml not found. Please ensure you're in the project directory."
        return 1
    fi
}

# Check Xcode Command Line Tools
check_xcode_tools() {
    print_status "Checking Xcode Command Line Tools..."
    if ! xcode-select -p &> /dev/null; then
        print_warning "Xcode Command Line Tools not found. Installing..."
        xcode-select --install
        echo "Please complete the Xcode Command Line Tools installation and run this script again."
        exit 1
    else
        print_success "Xcode Command Line Tools are installed"
    fi
}

# Create directories
create_directories() {
    print_status "Creating project directories..."
    mkdir -p test_images
    mkdir -p build
    print_success "Directories created"
}

# Check LibTorch installation
check_libtorch() {
    print_status "Checking LibTorch installation..."
    if [ ! -d "$HOME/libs/libtorch" ]; then
        print_warning "LibTorch not found in ~/libs/libtorch"
        echo "Please download LibTorch from: https://pytorch.org/get-started/locally/"
        echo "Choose: C++ -> LibTorch -> macOS -> CPU"
        echo "Extract to: ~/libs/libtorch/"
        echo ""
        echo "After downloading, run this script again."
        exit 1
    else
        print_success "LibTorch found in ~/libs/libtorch"
    fi
}

# Build the project
build_project() {
    print_status "Building the project..."
    cd build
    
    # Configure with CMake
    cmake -DCMAKE_PREFIX_PATH="$HOME/libs/libtorch" ..
    
    # Build
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    cd ..
    print_success "Project built successfully"
}

# Create environment setup script
create_env_script() {
    print_status "Creating environment setup script..."
    cat > setup_env.sh << 'EOF'
#!/bin/bash
# Environment setup script for shape recognizer

# Set LibTorch library path
export DYLD_LIBRARY_PATH="$HOME/libs/libtorch/lib:$DYLD_LIBRARY_PATH"

echo "Environment variables set:"
echo "DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
echo ""
echo "You can now run: ./build/shape_recognizer <image_path> [model_path]"
EOF

    chmod +x setup_env.sh
    print_success "Environment setup script created: setup_env.sh"
}

# Create a sample test image (if ImageMagick is available)
create_sample_image() {
    print_status "Creating sample test image..."
    if command -v convert &> /dev/null; then
        convert -size 100x100 xc:white -fill black -draw "circle 50,50 50,10" test_images/circle.png
        print_success "Sample circle image created: test_images/circle.png"
    else
        print_warning "ImageMagick not found. Please create test images manually in test_images/ directory"
    fi
}

# Main setup process
main() {
    echo "Starting setup process..."
    echo ""
    
    check_homebrew
    install_opencv
    check_poetry
    install_python_deps
    check_xcode_tools
    create_directories
    check_libtorch
    build_project
    create_env_script
    create_sample_image
    
    echo ""
    print_success "Setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Source the environment: source setup_env.sh"
    echo "2. Train and export your PyTorch model: poetry run train-model"
    echo "3. Add test images to test_images/ directory"
    echo "4. Run: ./build/shape_recognizer test_images/circle.png shape_model.pt"
    echo ""
    echo "For training a model, use: poetry run train-model"
}

# Run main function
main "$@" 