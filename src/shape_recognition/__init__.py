"""
Shape Recognition Package

A C++ application for shape recognition using PyTorch (LibTorch) and OpenCV,
specifically configured for macOS with Poetry for dependency management.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .train_model import ShapeClassifier, ShapeDataset, train_model, export_model
from .test_setup import test_import, test_command, test_file_exists

__all__ = [
    "ShapeClassifier",
    "ShapeDataset", 
    "train_model",
    "export_model",
    "test_import",
    "test_command", 
    "test_file_exists"
] 