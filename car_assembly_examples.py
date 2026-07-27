"""
Car Assembly Dataset - Example Usage and Testing Scripts

This file contains practical examples for:
1. Loading the dataset
2. Exploring the dataset
3. Training a classifier
4. Making predictions
5. Evaluating performance
"""

# ============================================================================
# 1. LOADING AND EXPLORING THE DATASET
# ============================================================================

"""
Example 1: Load dataset using PyTorch ImageFolder
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transforms
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder("car_assembly_dataset/train", 
                                     transform=transform)
val_dataset = datasets.ImageFolder("car_assembly_dataset/val", 
                                   transform=transform)
test_dataset = datasets.ImageFolder("car_assembly_dataset/test", 
                                    transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")


# ============================================================================
# Example 2: Load dataset using pandas and CSV files
# ============================================================================

import pandas as pd
from PIL import Image
import numpy as np

# Load metadata
train_df = pd.read_csv("car_assembly_dataset/annotations/train.csv")
val_df = pd.read_csv("car_assembly_dataset/annotations/val.csv")
test_df = pd.read_csv("car_assembly_dataset/annotations/test.csv")

print("Train DataFrame:")
print(train_df.head())
print(f"\nDataset statistics:")
print(train_df['damage_level'].value_counts())
print(f"\nAngles distribution:")
print(train_df['angle'].value_counts())


# ============================================================================
# Example 3: Visualize sample images
# ============================================================================

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def plot_images_from_loader(loader, num_images=4):
    """Plot sample images from data loader"""
    images, labels = next(iter(loader))
    
    # Denormalize for visualization
    denorm = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                           std=[1/0.229, 1/0.224, 1/0.225])
    ])
    
    images = denorm(images[:num_images])
    
    fig, axes = plt.subplots(1, num_images, figsize=(15, 4))
    class_names = ["highly_damaged", "medium_damaged", "low_damaged"]
    
    for idx in range(num_images):
        img = images[idx].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[idx].imshow(img)
        axes[idx].set_title(class_names[labels[idx].item()])
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

# Usage
plot_images_from_loader(train_loader, num_images=4)


# ============================================================================
# 2. CUSTOM DATASET CLASS
# ============================================================================

from torch.utils.data import Dataset
from pathlib import Path

class CarAssemblyDataset(Dataset):
    """Custom dataset for car assembly images with angle and damage info"""
    
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (str): Path to csv file with annotations
            root_dir (str): Directory with all images
            transform (callable): Optional transform to be applied
        """
        self.annotations_df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Create class mapping
        self.class_to_idx = {
            'highly_damaged': 0,
            'medium_damaged': 1,
            'low_damaged': 2
        }
    
    def __len__(self):
        return len(self.annotations_df)
    
    def __getitem__(self, idx):
        # Get image path and load
        row = self.annotations_df.iloc[idx]
        img_path = self.root_dir / row['image_path']
        image = Image.open(str(img_path)).convert('RGB')
        
        # Get label
        label = self.class_to_idx[row['damage_level']]
        
        # Get metadata
        angle = row['angle']
        car_id = row['car_id']
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'damage_level': row['damage_level'],
            'angle': angle,
            'car_id': car_id
        }

# Usage
custom_train_dataset = CarAssemblyDataset(
    csv_file="car_assembly_dataset/annotations/train.csv",
    root_dir="car_assembly_dataset",
    transform=transform
)

custom_loader = DataLoader(custom_train_dataset, batch_size=4, shuffle=True)


# ============================================================================
# 3. DATA STATISTICS AND ANALYSIS
# ============================================================================

def analyze_dataset(dataset_dir="car_assembly_dataset"):
    """Analyze and print dataset statistics"""
    
    metadata_path = Path(dataset_dir) / "metadata.json"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    images = metadata['images']
    
    # Count by damage level
    damage_counts = {}
    for img in images:
        damage = img['damage_level']
        damage_counts[damage] = damage_counts.get(damage, 0) + 1
    
    # Count by angle
    angle_counts = {}
    for img in images:
        angle = img['angle']
        angle_counts[angle] = angle_counts.get(angle, 0) + 1
    
    # Count by split
    split_counts = {}
    for img in images:
        split = img['split']
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print("="*50)
    print("DATASET ANALYSIS")
    print("="*50)
    print(f"\nTotal images: {len(images)}")
    print(f"\nDamage Level Distribution:")
    for damage, count in sorted(damage_counts.items()):
        print(f"  {damage}: {count} images")
    
    print(f"\nViewing Angle Distribution:")
    for angle, count in sorted(angle_counts.items()):
        print(f"  {angle}: {count} images")
    
    print(f"\nTrain/Val/Test Split:")
    for split, count in sorted(split_counts.items()):
        pct = 100 * count / len(images)
        print(f"  {split}: {count} images ({pct:.1f}%)")
    
    print("\n" + "="*50)

# Usage
import json
analyze_dataset()


# ============================================================================
# 4. QUICK CLASSIFIER TRAINING
# ============================================================================

import torch
import torch.nn as nn
from torchvision.models import resnet50

def train_classifier():
    """Quick training script for car damage classifier"""
    
    # Configuration
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_epochs': 30,
        'batch_size': 4,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
    }
    
    print(f"Using device: {config['device']}")
    
    # Create data loaders
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder("car_assembly_dataset/train",
                                        transform=train_transform)
    val_dataset = datasets.ImageFolder("car_assembly_dataset/val",
                                      transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False)
    
    # Create model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 3)  # 3 damage classes
    model = model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                lr=config['learning_rate'],
                                weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in train_loader:
            images = images.to(config['device'])
            labels = labels.to(config['device'])
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= len(train_dataset)
        train_acc = 100 * train_correct / len(train_dataset)
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(config['device'])
                labels = labels.to(config['device'])
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= len(val_dataset)
        val_acc = 100 * val_correct / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | "
              f"Train Loss: {train_loss:.4f} ({train_acc:.1f}%) | "
              f"Val Loss: {val_loss:.4f} ({val_acc:.1f}%)")
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_damage_classifier.pth")
            print(f"  ✓ Model saved!")
    
    return model

# Usage (uncomment to run)
# model = train_classifier()


# ============================================================================
# 5. INFERENCE / PREDICTION
# ============================================================================

def predict_damage_level(image_path, model_path="best_damage_classifier.pth"):
    """Predict damage level for a single image"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_names = ["highly_damaged", "medium_damaged", "low_damaged"]
    
    # Load model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return {
        'predicted_class': class_names[predicted_class],
        'confidence': confidence,
        'probabilities': {
            class_names[i]: probabilities[0, i].item()
            for i in range(len(class_names))
        }
    }

# Usage
# result = predict_damage_level("car_assembly_dataset/train/low_damaged/image.jpg")
# print(f"Prediction: {result['predicted_class']}")
# print(f"Confidence: {result['confidence']:.2%}")
# print(f"Probabilities: {result['probabilities']}")


# ============================================================================
# 6. BATCH PREDICTION ON TEST SET
# ============================================================================

def evaluate_on_test_set(model_path="best_damage_classifier.pth"):
    """Evaluate model on test set"""
    
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_names = ["highly_damaged", "medium_damaged", "low_damaged"]
    
    # Load test data
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder("car_assembly_dataset/test",
                                       transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Load model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get predictions
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Print report
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

# Usage
# evaluate_on_test_set()


# ============================================================================
# 7. ANALYZE BY ANGLE
# ============================================================================

def analyze_performance_by_angle(model_path="best_damage_classifier.pth"):
    """Analyze model performance for each viewing angle"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_names = ["highly_damaged", "medium_damaged", "low_damaged"]
    
    # Load custom dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = CarAssemblyDataset(
        csv_file="car_assembly_dataset/annotations/test.csv",
        root_dir="car_assembly_dataset",
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Load model
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Analyze by angle
    angle_stats = {}
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            angles = batch['angle']
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(len(angles)):
                angle = angles[i]
                is_correct = (predicted[i] == labels[i]).item()
                
                if angle not in angle_stats:
                    angle_stats[angle] = {'correct': 0, 'total': 0}
                
                angle_stats[angle]['total'] += 1
                if is_correct:
                    angle_stats[angle]['correct'] += 1
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE BY VIEWING ANGLE")
    print("="*60)
    
    for angle, stats in sorted(angle_stats.items()):
        accuracy = 100 * stats['correct'] / stats['total']
        print(f"{angle:10s}: {accuracy:.1f}% "
              f"({stats['correct']}/{stats['total']} correct)")

# Usage
# analyze_performance_by_angle()


if __name__ == "__main__":
    print("Car Assembly Dataset - Example Scripts")
    print("Uncomment the usage examples to run them!")
