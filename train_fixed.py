# Fixed Property Classification Training Script
# Addresses Windows multiprocessing and encoding issues

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
import multiprocessing

def create_dataloaders():
    """Create data loaders with proper Windows settings"""
    # Data directories
    DATA_DIR = "./Data"
    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "test")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_transform)
    
    # DataLoaders - use num_workers=0 for Windows
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, train_dataset.classes

def create_model(num_classes):
    """Create EfficientNet model"""
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes)  # EfficientNet-B0 has 1280 features
    )
    return model

def train_model():
    """Main training function"""
    # Configuration
    NUM_EPOCHS = 20
    FINE_TUNE_EPOCHS = 10
    LEARNING_RATE = 1e-4
    FINE_TUNE_LR = 1e-5
    TARGET_ACCURACY = 0.90
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Create models directory
    MODELS_DIR = "./models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Model save paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_SAVE_PATH = os.path.join(MODELS_DIR, f"best_model_{timestamp}.pth")
    TARGET_MODEL_PATH = os.path.join(MODELS_DIR, f"target_95_model_{timestamp}.pth")
    
    # Load data
    print("📁 Loading datasets...")
    train_loader, val_loader, class_names = create_dataloaders()
    num_classes = len(class_names)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Classes ({num_classes}): {class_names}")
    print(f"   Training samples: {len(train_loader.dataset)}")
    print(f"   Validation samples: {len(val_loader.dataset)}")
    print(f"   Target accuracy: {TARGET_ACCURACY*100}%")
    
    # Save class mapping
    class_mapping = {i: class_name for i, class_name in enumerate(class_names)}
    with open(os.path.join(MODELS_DIR, f"class_mapping_{timestamp}.json"), 'w', encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2, ensure_ascii=False)
    
    # Compute class weights
    train_labels = [label for _, label in train_loader.dataset]
    class_weights = compute_class_weight(class_weight='balanced', 
                                       classes=np.arange(num_classes), 
                                       y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("⚖️ Class weights computed for balanced training")
    
    # Create model
    model = create_model(num_classes)
    model = model.to(device)
    
    # Freeze base model initially
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Setup training
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training variables
    best_val_acc = 0.0
    target_reached = False
    training_history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'epochs': []}    
    print(f"\n🚀 Starting Initial Training Phase...")
    print("=" * 60)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"🏋️ Epoch {epoch+1}/{NUM_EPOCHS} [Training]", 
                         ncols=100, leave=False)
        
        for inputs, labels in train_pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size(0)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_size
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()
            train_samples += batch_size
            
            current_acc = running_corrects / train_samples
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'GPU': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'CPU'
            })
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_corrects = 0
        val_samples = 0
        
        val_pbar = tqdm(val_loader, desc=f"🔍 Epoch {epoch+1}/{NUM_EPOCHS} [Validation]", 
                       ncols=100, leave=False)
        
        with torch.no_grad():
            for inputs, labels in val_pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                batch_size = inputs.size(0)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data).item()
                val_samples += batch_size
                
                current_val_acc = val_corrects / val_samples
                val_pbar.set_postfix({
                    'Val_Acc': f'{current_val_acc:.4f}',
                    'Best': f'{best_val_acc:.4f}'
                })
        
        val_acc = val_corrects / len(val_loader.dataset)
        epoch_time = time.time() - epoch_start_time
        
        # Store history
        training_history['train_loss'].append(epoch_loss)
        training_history['train_acc'].append(epoch_acc)
        training_history['val_acc'].append(val_acc)
        training_history['epochs'].append(epoch + 1)
        
        # Enhanced epoch summary
        current_best = max(best_val_acc, val_acc)
        print(f"\n📈 Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
        print(f"   ⏱️  Time: {epoch_time:.2f}s")
        print(f"   📉 Train Loss: {epoch_loss:.6f}")
        print(f"   📊 Train Acc:  {epoch_acc:.4f} ({epoch_acc*100:.2f}%)")
        print(f"   🎯 Val Acc:    {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"   🏆 Best Val:   {current_best:.4f} ({current_best*100:.2f}%)")
        
        # Check if target accuracy reached
        if val_acc >= TARGET_ACCURACY and not target_reached:
            target_reached = True
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'train_acc': epoch_acc,
                'class_mapping': class_mapping
            }, TARGET_MODEL_PATH)
            print(f"\n🎉🎉🎉 TARGET ACHIEVED! 🎉🎉🎉")
            print(f"   🎯 Reached {val_acc*100:.2f}% accuracy at epoch {epoch+1}!")
            print(f"   💾 Model saved to: {TARGET_MODEL_PATH}")
            print(f"   🛑 Stopping training as target reached!")
            break
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_acc': val_acc,
                'train_acc': epoch_acc,
                'class_mapping': class_mapping
            }, MODEL_SAVE_PATH)
            print(f"   ✅ New best model saved! Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
        
        print("   " + "=" * 50)
    
    # Fine-tuning (if target not reached)
    if not target_reached and NUM_EPOCHS > 0:
        print(f"\n🔧 Starting Fine-tuning Phase...")
        print("=" * 60)
        
        # Unfreeze layers
        for param in model.features.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
        
        for epoch in range(FINE_TUNE_EPOCHS):
            # Similar training loop for fine-tuning
            # (Abbreviated for space - same structure as above)
            model.train()
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
            
            epoch_acc = running_corrects / len(train_loader.dataset)
            
            # Validation
            model.eval()
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data).item()
            
            val_acc = val_corrects / len(val_loader.dataset)
            
            print(f"🔧 Fine-tune {epoch+1}/{FINE_TUNE_EPOCHS}: "
                  f"Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            if val_acc >= TARGET_ACCURACY and not target_reached:
                target_reached = True
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': NUM_EPOCHS + epoch + 1,
                    'val_acc': val_acc,
                    'train_acc': epoch_acc,
                    'class_mapping': class_mapping
                }, TARGET_MODEL_PATH)
                print(f"🎉 TARGET ACHIEVED IN FINE-TUNING! 🎉")
                break
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': NUM_EPOCHS + epoch + 1,
                    'val_acc': val_acc,
                    'train_acc': epoch_acc,
                    'class_mapping': class_mapping
                }, MODEL_SAVE_PATH)
    
    # Final summary
    print(f"\n{'='*60}")
    print("🏁 TRAINING COMPLETE! 🏁")
    print(f"{'='*60}")
    print(f"📊 Final Results:")
    print(f"   🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   🎯 Target Accuracy: {TARGET_ACCURACY:.4f} ({TARGET_ACCURACY*100:.2f}%)")
    print(f"   ✅ Target Reached: {'YES' if target_reached else 'NO'}")
    print(f"\n💾 Saved Models:")
    print(f"   📁 Best Model: {MODEL_SAVE_PATH}")
    if target_reached:
        print(f"   🎯 Target Model: {TARGET_MODEL_PATH}")
    
    # Create training plot
    if training_history['epochs']:
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.plot(training_history['epochs'], training_history['train_loss'], 'b-', linewidth=2)
        plt.title('📉 Training Loss', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(training_history['epochs'], [acc*100 for acc in training_history['train_acc']], 'g-', linewidth=2)
        plt.title('📊 Training Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 3)
        plt.plot(training_history['epochs'], [acc*100 for acc in training_history['val_acc']], 'r-', linewidth=2)
        plt.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='Target (95%)')
        plt.title('🎯 Validation Accuracy', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(MODELS_DIR, f'training_history_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"   📈 Training Plot: {plot_path}")
        plt.show()
    
    print(f"\n🎉 All done! Your model is ready for inference.")
    if target_reached:
        print(f"🌟 Congratulations! You achieved your target accuracy of {TARGET_ACCURACY*100}%!")
    else:
        print(f"💡 Consider training for more epochs to reach {TARGET_ACCURACY*100}% accuracy.")

if __name__ == '__main__':
    # This guard is essential for Windows multiprocessing
    multiprocessing.freeze_support()
    train_model()