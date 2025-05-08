import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Enhanced Data Loading with Augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
    #transforms.RandomErasing(p=0.3, scale=(0.02, 0.1), ratio=(0.3, 2.0), value='random'),
    #transforms.RandomAffine(degrees=5, translate=(0.1, 0.1)),
])

# Transform for VALIDATION set (no augmentation)
transform_val_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])

# Load datasets
training_data = torchvision.datasets.CIFAR10('./', download=True, train=True)
test_data = torchvision.datasets.CIFAR10('./', download=True, train=False, transform=transform_val_test)

# Split into train and validation
train_data, val_data = random_split(training_data, [40000, 10000])

# Apply augmentation only to training set
train_data.dataset.transform = transform_train

# Validation set uses minimal transforms
val_data.dataset.transform = transform_val_test

# Data loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size, num_workers=4)
test_loader = DataLoader(test_data, batch_size, num_workers=2)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Conv Block 1 (32x32x3 → 16x16x64)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Conv Block 2 (16x16x64 → 8x8x128)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Conv Block 3 (8x8x128 → 4x4x256)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Global Average Pooling (4x4x256 → 1x1x256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classifier
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 32x32 → 16x16
        
        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 16x16 → 8x8
        
        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)  # 8x8 → 4x4
        
        # Global Pooling + Flatten
        x = self.gap(x)  # 4x4x256 → 1x1x256
        x = torch.flatten(x, 1)
        
        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
# Training loop
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    train_loss = running_loss / len(loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(loader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


model = CIFAR10Model()


epochs = 30

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
criterion = nn.CrossEntropyLoss()#label_smoothing=0.1)


# Main training process

best_acc = 0.0
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    
    scheduler.step(val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}:')
    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
    if epoch % 5 == 0:
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%')
    
    
    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# Load best model and test
model.load_state_dict(torch.load('best_model.pth'))
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f'Val Loss: {test_loss:.4f} |Test Accuracy: {test_acc:.2f}%')