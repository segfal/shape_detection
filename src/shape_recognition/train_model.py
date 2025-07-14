#!/usr/bin/env python3
"""
Shape Recognition Model Trainer
Trains a PyTorch model for shape classification and exports it for C++ use.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import random
from pathlib import Path

class ShapeDataset(Dataset):
    """Custom dataset for generating synthetic shape images."""
    
    def __init__(self, num_samples=1000, image_size=28, transform=None):
        self.num_samples = num_samples
        self.image_size = image_size
        self.transform = transform
        self.shapes = ['circle', 'square', 'triangle', 'rectangle']
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Randomly select a shape
        shape_idx = random.randint(0, len(self.shapes) - 1)
        shape = self.shapes[shape_idx]
        
        # Create image
        image = self.create_shape_image(shape)
        
        if self.transform:
            image = self.transform(image)
        
        return image, shape_idx
    
    def create_shape_image(self, shape):
        """Create a synthetic shape image."""
        # Create white background
        image = Image.new('L', (self.image_size, self.image_size), 255)
        draw = ImageDraw.Draw(image)
        
        # Define shape parameters
        margin = 4
        size = self.image_size - 2 * margin
        
        if shape == 'circle':
            # Draw circle
            draw.ellipse([margin, margin, margin + size, margin + size], 
                        outline=0, fill=0)
            
        elif shape == 'square':
            # Draw square
            draw.rectangle([margin, margin, margin + size, margin + size], 
                          outline=0, fill=0)
            
        elif shape == 'triangle':
            # Draw triangle
            points = [
                (self.image_size // 2, margin),
                (margin, margin + size),
                (margin + size, margin + size)
            ]
            draw.polygon(points, outline=0, fill=0)
            
        elif shape == 'rectangle':
            # Draw rectangle (2:1 aspect ratio)
            rect_width = size
            rect_height = size // 2
            x_offset = (size - rect_width) // 2
            y_offset = (size - rect_height) // 2
            draw.rectangle([
                margin + x_offset, 
                margin + y_offset, 
                margin + x_offset + rect_width, 
                margin + y_offset + rect_height
            ], outline=0, fill=0)
        
        return image

class ShapeClassifier(nn.Module):
    """CNN model for shape classification."""
    
    def __init__(self, num_classes=4):
        super(ShapeClassifier, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.pool(self.relu(self.conv3(x)))  # 7x7 -> 3x3
        
        # Flatten
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("🚀 Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Training Loss: {train_loss:.4f}')
        print(f'  Validation Loss: {val_loss:.4f}')
        print(f'  Validation Accuracy: {val_accuracy:.2f}%')
        print('-' * 50)
    
    return train_losses, val_losses, val_accuracies

def plot_training_history(train_losses, val_losses, val_accuracies, save_path='training_history.png'):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot losses
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def export_model(model, save_path='shape_model.pt'):
    """Export the model to TorchScript format for C++ use."""
    model.eval()
    
    # Create example input (1 channel, 28x28 image)
    example_input = torch.rand(1, 1, 28, 28)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    traced_model.save(save_path)
    print(f"✅ Model exported to: {save_path}")
    
    # Test the exported model
    loaded_model = torch.jit.load(save_path)
    test_output = loaded_model(example_input)
    print(f"✅ Exported model test successful. Output shape: {test_output.shape}")

def main():
    """Main training function."""
    print("🔍 Shape Recognition Model Trainer")
    print("==================================")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
    ])
    
    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = ShapeDataset(num_samples=2000, transform=transform)
    val_dataset = ShapeDataset(num_samples=500, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    print("🧠 Creating model...")
    model = ShapeClassifier(num_classes=4).to(device)
    
    # Print model summary
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train the model
    train_losses, val_losses, val_accuracies = train_model(
        model, train_loader, val_loader, num_epochs=15, device=device
    )
    
    # Plot training history
    print("📈 Plotting training history...")
    plot_training_history(train_losses, val_losses, val_accuracies)
    
    # Export model
    print("💾 Exporting model...")
    export_model(model, 'shape_model.pt')
    
    print("\n🎉 Training completed successfully!")
    print("Next steps:")
    print("1. Copy shape_model.pt to your C++ project directory")
    print("2. Run the C++ application with: ./shape_recognizer <image_path> shape_model.pt")

if __name__ == "__main__":
    main() 