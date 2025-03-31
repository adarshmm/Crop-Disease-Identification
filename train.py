import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import UnidentifiedImageError

class CropDiseaseDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        valid_pairs = []
        self.valid_image_paths = []
        self.valid_labels = []
        
        for img_path, label in zip(image_paths, labels):
            try:
                with Image.open(img_path) as img:
                    img.verify()
                    self.valid_image_paths.append(img_path)
                    self.valid_labels.append(label)
            except (UnidentifiedImageError, OSError, IOError) as e:
                print(f"Skipping corrupted image {img_path}: {str(e)}")
                continue
        
        self.transform = transform
        print(f"Successfully loaded {len(self.valid_image_paths)} valid images")

    def __len__(self):
        return len(self.valid_image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.valid_image_paths[idx]).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.valid_labels[idx]
            return image, label
        except Exception as e:
            print(f"Error loading image {self.valid_image_paths[idx]}: {str(e)}")
            raise e

def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
            img = Image.open(filepath)
            img.load()
        return True
    except Exception as e:
        print(f"Invalid image file {filepath}: {str(e)}")
        return False

def load_data(data_dir):
    image_paths = []
    labels = []
    label_to_idx = {}
    current_label = 0
    
    print("Scanning dataset directory...")
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                full_path = os.path.join(root, file)
                
                if not is_valid_image(full_path):
                    continue
                
                class_name = os.path.basename(root)
                if class_name not in label_to_idx:
                    label_to_idx[class_name] = current_label
                    current_label += 1
                
                image_paths.append(full_path)
                labels.append(label_to_idx[class_name])
    
    print("\nDataset Summary:")
    print(f"Total images found: {len(image_paths)}")
    print(f"Number of classes: {len(label_to_idx)}")
    print(f"Classes: {list(label_to_idx.keys())}")
    
    if len(image_paths) == 0:
        raise ValueError("No valid images found in the dataset directory!")
    
    return image_paths, labels, label_to_idx

def create_model(num_classes):
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 50 == 0:  # Print every 50 batches
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}]')
                print(f'Loss: {running_loss/(i+1):.4f}, Acc: {100.*correct/total:.2f}%')
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model

def main(data_dir, batch_size=32, num_epochs=10):
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found: {data_dir}")
    
    print(f"Loading data from: {data_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data augmentation and preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    try:
        image_paths, labels, label_to_idx = load_data(data_dir)
        
        X_train, X_val, y_train, y_val = train_test_split(
            image_paths, labels, test_size=0.2, random_state=42
        )
        
        train_dataset = CropDiseaseDataset(X_train, y_train, transform=transform)
        val_dataset = CropDiseaseDataset(X_val, y_val, transform=transform)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("No valid images found after creating datasets!")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create and train model
        num_classes = len(label_to_idx)
        model = create_model(num_classes)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
        
        # Train the model
        model = train_model(model, train_loader, val_loader, criterion, optimizer, 
                          num_epochs=num_epochs, device=device)
        
        return model, label_to_idx
        
    except Exception as e:
        print(f"Error during dataset preparation: {str(e)}")
        return None, None

def run_training(data_dir):
    try:
        print("Starting training process...")
        model, label_mapping = main(
            data_dir=data_dir,
            batch_size=32,
            num_epochs=10
        )
        
        if model is None:
            print("Training failed!")
            return
        
        print("Training completed successfully!")
        print("Label mapping:", label_mapping)
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")

# Use the code
if __name__ == "__main__":
    data_directory = "archive"  # Replace with your data path
    run_training(data_directory)