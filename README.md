# 🔍 Shape Recognizer with PyTorch (LibTorch) - macOS

A C++ application for shape recognition using PyTorch's LibTorch and OpenCV, specifically configured for macOS.

## 🚀 Quick Start

### Prerequisites

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install OpenCV**:
   ```bash
   brew install opencv
   ```

3. **Install Xcode Command Line Tools**:
   ```bash
   xcode-select --install
   ```

4. **Install Poetry** (Python dependency management):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

### Download LibTorch

1. Go to [PyTorch Get Started](https://pytorch.org/get-started/locally/)
2. Select:
   - **Language**: C++
   - **Package**: LibTorch
   - **OS**: macOS
   - **Compute**: CPU
3. Download and extract to `~/libs/libtorch/`

### Build and Run

```bash
# Clone or navigate to project directory
cd shape_detection

# Install Python dependencies with Poetry
poetry install

# Create build directory
mkdir build && cd build

# Configure with CMake (adjust path if needed)
cmake -DCMAKE_PREFIX_PATH=~/libs/libtorch ..

# Build
make

# Set library path and run
export DYLD_LIBRARY_PATH=~/libs/libtorch/lib:$DYLD_LIBRARY_PATH
./shape_recognizer ../test_images/circle.png ../shape_model.pt
```

## 📁 Project Structure

```
shape_detection/
├── CMakeLists.txt          # Build configuration
├── main.cpp               # Main application code
├── README.md              # This file
├── shape_model.pt         # Your trained PyTorch model (add this)
└── test_images/           # Test images directory (create this)
    └── circle.png         # Example test image
```

## 🐍 Training Your Model (Python)

The project includes a complete training script with Poetry dependency management:

```bash
# Install dependencies and train model
poetry install
poetry run train-model
```

Or manually:

```python
# Activate Poetry environment
poetry shell

# Run training script
python src/shape_recognition/train_model.py
```

The training script will:
- Generate synthetic shape datasets
- Train a CNN model for 4 shape classes
- Export the model to TorchScript format
- Create training visualization plots

## 🔧 Configuration

### Update CMakeLists.txt Paths

If your LibTorch is installed in a different location, update the path in `CMakeLists.txt`:

```cmake
set(CMAKE_PREFIX_PATH "/path/to/your/libtorch")
```

### OpenCV Paths

The CMakeLists.txt includes common OpenCV paths for macOS:
- Apple Silicon: `/opt/homebrew/opt/opencv`
- Intel Macs: `/usr/local/opt/opencv`

## 🐛 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| `dyld: Library not loaded: ...libc10.dylib` | Set `DYLD_LIBRARY_PATH=~/libs/libtorch/lib:$DYLD_LIBRARY_PATH` |
| OpenCV not found | Check Homebrew installation: `brew list opencv` |
| C++17 errors | Update Xcode tools: `xcode-select --install` |
| CMake can't find Torch | Verify LibTorch path in CMakeLists.txt |
| Poetry not found | Install with: `curl -sSL https://install.python-poetry.org \| python3 -` |
| Python dependencies missing | Run: `poetry install` |

### Debug Commands

```bash
# Check OpenCV installation
pkg-config --cflags --libs opencv4

# Check LibTorch installation
ls ~/libs/libtorch/lib/

# Verify library paths
otool -L ./shape_recognizer

# Check Poetry environment
poetry env info
poetry show
```

## 📝 Usage Examples

```bash
# Basic usage
./shape_recognizer image.png

# Specify model path
./shape_recognizer image.png custom_model.pt

# Test with different images
./shape_recognizer test_images/square.png
./shape_recognizer test_images/triangle.png
```

## 🔒 Security Considerations

- **Input Validation**: The application validates image files before processing
- **Error Handling**: Comprehensive exception handling prevents crashes
- **Memory Management**: Uses RAII and smart pointers where appropriate

## ⚡ Performance Notes

- **CPU Only**: This setup uses CPU inference (GPU not supported on macOS with PyTorch)
- **Memory Usage**: Images are resized to 28x28 to reduce memory footprint
- **Batch Processing**: The code supports single image inference; for batch processing, modify the tensor operations

## 🧪 Testing

Create test images in the `test_images/` directory:

```bash
mkdir test_images
# Add your test images here
```

## 📚 Resources

- [PyTorch C++ API Documentation](https://pytorch.org/cppdocs/)
- [OpenCV C++ Documentation](https://docs.opencv.org/)
- [CMake Documentation](https://cmake.org/documentation/)
- [LibTorch Installation Guide](https://pytorch.org/cppdocs/installing.html)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Note**: This project is specifically configured for macOS. For other platforms, you'll need to adjust the CMakeLists.txt and installation paths accordingly. 