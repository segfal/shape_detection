#!/usr/bin/env python3
"""
Test script to verify the setup for shape recognition.
This script checks if all required dependencies are available.
"""

import sys
import subprocess
import os
from pathlib import Path

def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name} - OK")
        return True
    except ImportError as e:
        print(f"❌ {package_name or module_name} - FAILED: {e}")
        return False

def test_command(command, description):
    """Test if a command is available."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - OK")
            return True
        else:
            print(f"❌ {description} - FAILED")
            return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def test_file_exists(file_path, description):
    """Test if a file exists."""
    if os.path.exists(file_path):
        print(f"✅ {description} - OK")
        return True
    else:
        print(f"❌ {description} - NOT FOUND")
        return False

def test_poetry_environment():
    """Test Poetry environment setup."""
    print("📦 Poetry Environment:")
    print("-" * 20)
    
    # Check if Poetry is installed
    if not test_command("poetry --version", "Poetry"):
        print("❌ Poetry not installed. Install with: curl -sSL https://install.python-poetry.org | python3 -")
        return False
    
    # Check if we're in a Poetry environment
    if os.environ.get('VIRTUAL_ENV') and 'poetry' in os.environ.get('VIRTUAL_ENV', ''):
        print("✅ Poetry virtual environment active")
    else:
        print("⚠️  Not in Poetry virtual environment. Run: poetry shell")
    
    # Check if pyproject.toml exists
    if not test_file_exists("pyproject.toml", "pyproject.toml"):
        print("❌ pyproject.toml not found")
        return False
    
    return True

def main():
    """Run all tests."""
    print("🔍 Shape Recognition Setup Test (Poetry)")
    print("========================================")
    print()
    
    all_tests_passed = True
    
    # Test Poetry environment
    if not test_poetry_environment():
        all_tests_passed = False
    
    print()
    
    # Test Python dependencies
    print("🐍 Python Dependencies:")
    print("-" * 20)
    
    python_deps = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("cv2", "OpenCV-Python"),
        ("PIL", "Pillow"),
        ("matplotlib", "Matplotlib"),
        ("numpy", "NumPy")
    ]
    
    for module, name in python_deps:
        if not test_import(module, name):
            all_tests_passed = False
    
    print()
    
    # Test system commands
    print("🛠️  System Commands:")
    print("-" * 20)
    
    system_commands = [
        ("brew --version", "Homebrew"),
        ("cmake --version", "CMake"),
        ("make --version", "Make"),
        ("g++ --version", "GCC/G++")
    ]
    
    for command, description in system_commands:
        if not test_command(command, description):
            all_tests_passed = False
    
    print()
    
    # Test OpenCV installation
    print("📷 OpenCV Installation:")
    print("-" * 20)
    
    opencv_paths = [
        ("/opt/homebrew/opt/opencv", "OpenCV (Apple Silicon)"),
        ("/usr/local/opt/opencv", "OpenCV (Intel)")
    ]
    
    opencv_found = False
    for path, description in opencv_paths:
        if os.path.exists(path):
            print(f"✅ {description} - Found at {path}")
            opencv_found = True
            break
    
    if not opencv_found:
        print("❌ OpenCV not found in common locations")
        all_tests_passed = False
    
    print()
    
    # Test LibTorch installation
    print("🧠 LibTorch Installation:")
    print("-" * 20)
    
    libtorch_paths = [
        os.path.expanduser("~/libs/libtorch"),
        "/usr/local/libtorch",
        "./libtorch"
    ]
    
    libtorch_found = False
    for path in libtorch_paths:
        if os.path.exists(path):
            print(f"✅ LibTorch found at: {path}")
            libtorch_found = True
            break
    
    if not libtorch_found:
        print("❌ LibTorch not found")
        print("   Please download from: https://pytorch.org/get-started/locally/")
        print("   Extract to ~/libs/libtorch/")
        all_tests_passed = False
    
    print()
    
    # Test project files
    print("📁 Project Files:")
    print("-" * 20)
    
    project_files = [
        ("CMakeLists.txt", "CMakeLists.txt"),
        ("main.cpp", "main.cpp"),
        ("setup.sh", "setup.sh"),
        ("pyproject.toml", "pyproject.toml"),
        ("src/shape_recognition/__init__.py", "Package __init__.py")
    ]
    
    for file_path, description in project_files:
        if not test_file_exists(file_path, description):
            all_tests_passed = False
    
    print()
    
    # Summary
    print("📊 Test Summary:")
    print("-" * 20)
    
    if all_tests_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print()
        print("Next steps:")
        print("1. Run: poetry run train-model")
        print("2. Run: ./setup.sh")
        print("3. Test: ./build/shape_recognizer test_images/circle.png shape_model.pt")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
        print()
        print("Common fixes:")
        print("- Install Poetry: curl -sSL https://install.python-poetry.org | python3 -")
        print("- Install dependencies: poetry install")
        print("- Activate environment: poetry shell")
        print("- Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
        print("- Install OpenCV: brew install opencv")
        print("- Download LibTorch from: https://pytorch.org/get-started/locally/")
        print("- Install Xcode tools: xcode-select --install")

if __name__ == "__main__":
    main() 