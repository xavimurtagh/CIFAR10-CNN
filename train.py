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
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load datasets
train_data = torchvision.datasets.CIFAR10('./', download=True, train=True, transform=transform_train)
test_data = torchvision.datasets.CIFAR10('./', download=True, train=False, transform=transform_test)

# Split into train and validation
train_data, val_data = random_split(train_data, [40000, 10000])

# Data loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size, num_workers=4)
test_loader = DataLoader(test_data, batch_size, num_workers=2)

class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()      
        self.conv1 = nn.Conv2d(3, 12, 5) 
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5) 
        
        self.dropout = nn.Dropout(0.2)

        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) 
        x = self.pool(F.relu(self.conv2(x))) 
        
        #x = self.pool(F.relu(self.conv3(x))) # 128 * 6 * 6
        #x = F.relu(self.conv4(x)) # -> 128 * 8 * 8
        #x = self.pool(F.relu(self.conv5(x))) # 256 * 3 * 3
        #x = self.pool(F.relu(self.conv6(x))) # -> 128 * 5 * 5
        
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = self.fc3(x)

        return x
    
net = CIFAR10Model()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

for epoch in range(10):
    print(f'Epoch {epoch}:')
    running_loss = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f'Loss: {running_loss / len(train_loader):.4f}')
    
correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct/total
print(f'Test Accuracy: {accuracy:.2f}%')


optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(12):
    print(f'Epoch {epoch}:')
    running_loss = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f'Loss: {running_loss / len(train_loader):.4f}')
    
correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct/total
print(f'Test Accuracy: {accuracy:.2f}%')


optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

for epoch in range(10):
    print(f'Epoch {epoch}:')
    running_loss = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f'Loss: {running_loss / len(train_loader):.4f}')
    
correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct/total
print(f'Test Accuracy: {accuracy:.2f}%')


optimizer = torch.optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)

for epoch in range(5):
    print(f'Epoch {epoch}:')
    running_loss = 0
    
    for i, data in enumerate(train_loader):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f'Loss: {running_loss / len(train_loader):.4f}')


correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct/total
print(f'Test Accuracy: {accuracy:.2f}%')

with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct/total
print(f'Test Accuracy: {accuracy:.2f}%')


correct = 0
total = 0

net.eval()

with torch.no_grad():
    for data in train_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct/total
print(f'Test Accuracy: {accuracy:.2f}%')