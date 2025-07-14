# 🚀 Quick Start Guide - Shape Recognizer

Get your C++ shape recognizer with PyTorch running in 5 minutes!

## ⚡ Super Quick Setup

### 1. Install Dependencies (One-time setup)

```bash
# Install Homebrew (if needed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install OpenCV
brew install opencv

# Install Xcode tools (if needed)
xcode-select --install

# Install Poetry (Python dependency management)
curl -sSL https://install.python-poetry.org | python3 -
```

### 2. Download LibTorch

1. Go to: https://pytorch.org/get-started/locally/
2. Select: **C++** → **LibTorch** → **macOS** → **CPU**
3. Download and extract to: `~/libs/libtorch/`

### 3. Test Your Setup

```bash
# Run the test script
python test_setup.py
```

### 4. Train a Model

```bash
# Install Python dependencies with Poetry
poetry install

# Train and export the model
poetry run train-model
```

### 5. Build and Run

```bash
# Run the automated setup (includes Poetry setup)
./setup.sh

# Or build manually
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=~/libs/libtorch ..
make

# Set environment and run
export DYLD_LIBRARY_PATH=~/libs/libtorch/lib:$DYLD_LIBRARY_PATH
./shape_recognizer ../test_images/circle.png ../shape_model.pt
```

## 🎯 What You'll Get

- ✅ **C++ Application**: Fast shape recognition using LibTorch
- ✅ **Trained Model**: CNN model for 4 shape classes (circle, square, triangle, rectangle)
- ✅ **OpenCV Integration**: Image loading, preprocessing, and display
- ✅ **macOS Optimized**: Configured specifically for your Mac

## 🔧 Troubleshooting

### Common Issues

| Problem | Quick Fix |
|---------|-----------|
| `dyld: Library not loaded` | `export DYLD_LIBRARY_PATH=~/libs/libtorch/lib:$DYLD_LIBRARY_PATH` |
| OpenCV not found | `brew install opencv` |
| CMake errors | `xcode-select --install` |
| Poetry not found | `curl -sSL https://install.python-poetry.org \| python3 -` |
| Python import errors | `poetry install` |

### Need Help?

1. Run `poetry run test-setup` to diagnose issues
2. Check the full [README.md](README.md) for detailed instructions
3. Verify your LibTorch path in `CMakeLists.txt`

## 📁 Project Structure

```
shape_detection/
├── main.cpp              # C++ application
├── CMakeLists.txt        # Build configuration
├── pyproject.toml        # Poetry configuration
├── setup.sh             # Automated setup script
├── src/shape_recognition/ # Python package
│   ├── __init__.py
│   ├── train_model.py    # Model trainer
│   └── test_setup.py     # Dependency checker
├── shape_model.pt       # Trained model (generated)
└── test_images/         # Test images directory
```

## 🎉 Success!

Once running, you'll see:
- Image loading confirmation
- Shape prediction results
- Visual display with prediction overlay

**Example output:**
```
🔍 Shape Recognizer with PyTorch (LibTorch)
==========================================
📸 Loaded image: test_images/circle.png
   Size: 100x100
✅ Model loaded successfully from: shape_model.pt
🎯 Prediction: circle
```

---

**Ready to customize?** Check out the full [README.md](README.md) for advanced features and configuration options! 