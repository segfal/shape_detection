# 🔍 Shape Recognition with PyTorch (LibTorch) - macOS

A complete C++/Python hybrid application for real-time shape recognition using PyTorch's LibTorch and OpenCV, specifically optimized for macOS with Poetry dependency management.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Key Functions](#key-functions)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## 🎯 Project Overview

This project demonstrates a hybrid approach to machine learning deployment:
- **Python**: Model training, data generation, and experimentation
- **C++**: High-performance inference with LibTorch for production deployment
- **Poetry**: Modern Python dependency management
- **CMake**: Cross-platform C++ build system

### Supported Shapes
- Circle
- Square  
- Triangle
- Rectangle

## 🏗️ Architecture

```
shape_detection/
├── 📁 src/shape_recognition/     # Python Package
│   ├── __init__.py              # Package initialization
│   ├── train_model.py           # Model training & export
│   └── test_setup.py            # Environment verification
├── 📄 main.cpp                  # C++ Application (Inference)
├── 📄 CMakeLists.txt            # C++ Build Configuration
├── 📄 pyproject.toml            # Poetry Configuration
├── 📄 setup.sh                  # Automated Setup Script
├── 📄 download_libtorch.sh      # LibTorch Downloader
├── 📄 install_poetry.sh         # Poetry Installation
└── 📁 test_images/              # Test Images Directory
```

### Data Flow
```
Python Training → TorchScript Export → C++ Inference → Real-time Results
```

## 🔧 Key Functions

### Python Functions (`src/shape_recognition/`)

#### `ShapeDataset` Class
**Purpose**: Synthetic dataset generation for training
```python
class ShapeDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=28, transform=None)
    def create_shape_image(self, shape: str) -> PIL.Image
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]
```

**Key Features**:
- Generates synthetic shape images programmatically
- Supports 4 shape classes: circle, square, triangle, rectangle
- Configurable image size and transformations
- Memory-efficient dataset generation

#### `ShapeClassifier` Class
**Purpose**: CNN model architecture for shape classification
```python
class ShapeClassifier(nn.Module):
    def __init__(self, num_classes=4)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```

**Architecture**:
- **Convolutional Layers**: 3 conv layers (32→64→128 channels)
- **Pooling**: MaxPool2d for dimension reduction
- **Fully Connected**: 256 hidden units → 4 output classes
- **Activation**: ReLU with Dropout (0.25)

#### Training Functions
```python
def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu')
def plot_training_history(train_losses, val_losses, val_accuracies)
def export_model(model, save_path='shape_model.pt')
```

**Training Process**:
- Adam optimizer with learning rate 0.001
- CrossEntropyLoss for multi-class classification
- Validation accuracy tracking
- Automatic model export to TorchScript format

### C++ Functions (`main.cpp`)

#### `ShapeRecognizer` Class
**Purpose**: High-performance inference engine
```cpp
class ShapeRecognizer {
    ShapeRecognizer(const std::string& model_path);
    std::string predictShape(const cv::Mat& image);
private:
    cv::Mat preprocessImage(const cv::Mat& image);
    torch::Tensor cvMatToTensor(const cv::Mat& image);
    std::string getShapeName(int class_id);
};
```

#### Key C++ Functions

**`preprocessImage()`**
```cpp
cv::Mat preprocessImage(const cv::Mat& image) {
    // 1. Convert to grayscale
    // 2. Resize to 28x28
    // 3. Normalize to [0, 1]
    // 4. Convert to CV_32F format
}
```

**`cvMatToTensor()`**
```cpp
torch::Tensor cvMatToTensor(const cv::Mat& image) {
    // Convert OpenCV Mat to PyTorch Tensor
    // Add channel dimension for grayscale
    // Ensure proper memory layout
}
```

**`predictShape()`**
```cpp
std::string predictShape(const cv::Mat& image) {
    // 1. Preprocess input image
    // 2. Convert to tensor
    // 3. Add batch dimension
    // 4. Run inference
    // 5. Extract prediction
    // 6. Return shape name
}
```

## 🚀 Installation

### Prerequisites
- macOS (Apple Silicon or Intel)
- Xcode Command Line Tools
- Homebrew

### Quick Installation
```bash
# 1. Clone the repository
git clone <your-repo-url>
cd shape_detection

# 2. Install Poetry and dependencies
./install_poetry.sh

# 3. Download LibTorch
./download_libtorch.sh

# 4. Build C++ application
./setup.sh
```

### Manual Installation
```bash
# Install system dependencies
brew install opencv cmake

# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install Python dependencies
poetry install

# Download LibTorch manually
# Visit: https://pytorch.org/get-started/locally/
# Extract to ~/libs/libtorch/
```

## 📖 Usage

### 1. Training the Model
```bash
# Activate Poetry environment
poetry shell

# Train and export model
poetry run train-model

# Or run directly
python src/shape_recognition/train_model.py
```

**Training Output**:
```
🔍 Shape Recognition Model Trainer
==================================
📊 Creating datasets...
🧠 Creating model...
Model parameters: 388,868
🚀 Starting training...
✅ Model exported to: shape_model.pt
```

### 2. Running C++ Inference
```bash
# Set up environment
source setup_env.sh

# Run shape recognition
./build/shape_recognizer <image_path> [model_path]

# Examples
./build/shape_recognizer test_images/circle.png shape_model.pt
./build/shape_recognizer test_images/square.png shape_model.pt
```

**Inference Output**:
```
🔍 Shape Recognizer with PyTorch (LibTorch)
==========================================
📸 Loaded image: test_images/circle.png
   Size: 100x100
✅ Model loaded successfully from: shape_model.pt
🎯 Prediction: circle
```

### 3. Testing Setup
```bash
# Verify all dependencies
poetry run test-setup

# Expected output: "🎉 All tests passed!"
```

## 🛠️ Development

### Project Structure Deep Dive

#### Python Package (`src/shape_recognition/`)
- **Modular Design**: Each component is a separate module
- **Type Hints**: Full type annotation for better IDE support
- **Error Handling**: Comprehensive exception handling
- **Documentation**: Detailed docstrings for all functions

#### C++ Application (`main.cpp`)
- **RAII**: Resource management with smart pointers
- **Exception Safety**: Comprehensive error handling
- **Memory Efficiency**: Minimal memory allocations
- **Performance**: Optimized for real-time inference

### Adding New Shapes

#### 1. Update Python Training
```python
# In ShapeDataset class
self.shapes = ['circle', 'square', 'triangle', 'rectangle', 'new_shape']

def create_shape_image(self, shape):
    if shape == 'new_shape':
        # Add drawing logic for new shape
        pass
```

#### 2. Update C++ Inference
```cpp
// In ShapeRecognizer class
std::string getShapeName(int class_id) {
    std::vector<std::string> shapes = {
        "circle", "square", "triangle", "rectangle", "new_shape"
    };
    // ...
}
```

#### 3. Retrain Model
```bash
poetry run train-model
```

### Customizing Model Architecture

#### Modify CNN Architecture
```python
class CustomShapeClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Add your custom layers here
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        # ... more layers
```

#### Adjust Training Parameters
```python
# In train_model.py
def train_model(model, train_loader, val_loader, num_epochs=20, device='cpu'):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjust learning rate
    # ... rest of training logic
```

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `dyld: Library not loaded` | Run `source setup_env.sh` |
| CMake version error | Update CMakeLists.txt version requirement |
| Poetry not found | Run `./install_poetry.sh` |
| LibTorch not found | Run `./download_libtorch.sh` |
| OpenCV not found | `brew install opencv` |
| Build failures | Check Xcode tools: `xcode-select --install` |

### Debug Commands
```bash
# Check environment
poetry run test-setup

# Verify LibTorch installation
ls ~/libs/libtorch/lib/

# Check OpenCV installation
pkg-config --cflags --libs opencv4

# Debug CMake
cd build && cmake -DCMAKE_PREFIX_PATH=~/libs/libtorch .. -DCMAKE_VERBOSE_MAKEFILE=ON
```

### Performance Optimization

#### C++ Optimization
```cpp
// Enable optimizations in CMakeLists.txt
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_BUILD_TYPE Release)
```

#### Python Optimization
```python
# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

## 📚 API Reference

### Python API

#### `ShapeDataset`
```python
dataset = ShapeDataset(
    num_samples=1000,    # Number of synthetic images
    image_size=28,       # Image dimensions
    transform=None       # Optional transforms
)
```

#### `ShapeClassifier`
```python
model = ShapeClassifier(num_classes=4)
output = model(input_tensor)  # Shape: [batch_size, num_classes]
```

#### `train_model()`
```python
train_losses, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, 
    num_epochs=15, device='cpu'
)
```

### C++ API

#### `ShapeRecognizer`
```cpp
ShapeRecognizer recognizer("shape_model.pt");
std::string prediction = recognizer.predictShape(image);
```

#### Main Function
```cpp
int main(int argc, char* argv[]) {
    // argc: number of arguments
    // argv[1]: image path
    // argv[2]: model path (optional)
}
```

## 🔒 Security Considerations

- **Input Validation**: All image inputs are validated before processing
- **Memory Management**: RAII ensures proper resource cleanup
- **Error Handling**: Comprehensive exception handling prevents crashes
- **Dependency Isolation**: Poetry virtual environments prevent conflicts

## ⚡ Performance Notes

- **CPU Only**: Optimized for macOS CPU inference (GPU not supported)
- **Memory Usage**: Images resized to 28x28 for efficiency
- **Inference Speed**: ~10ms per image on modern Macs
- **Model Size**: ~1.5MB TorchScript model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Install development dependencies
poetry install --with dev

# Run code formatting
poetry run black src/

# Run linting
poetry run flake8 src/

# Run type checking
poetry run mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for LibTorch
- OpenCV contributors
- Poetry team for dependency management
- CMake community for build system

---

**Ready to build amazing shape recognition applications? Start with this solid foundation! 🚀** 