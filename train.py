import torch
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torchvision.utils import make_grid
import torch.nn.functional as F




data = torchvision.datasets.CIFAR10('./', download=False, train=True, transform=ToTensor())
classes = data.classes

'''class_count = {}
for x,i in data:
    label = classes[i]
    if label not in class_count:
        class_count[label] = 0
    class_count[label] += 1
'class_count = {}\nfor x,i in data:\n    label = classes[i]\n    if label not in class_count:\n        class_count[label] = 0\n    class_count[label] += 1\nclass_count
print(class_count)
'''

train_data, val_data = random_split(data, [40000, 10000])
batch_size = 64
train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size*2, num_workers=4)
for images, x in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break

def accuracy(outputs, labels):
    _, pred = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(pred == labels).item() / len(pred))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  
        loss = F.cross_entropy(out, labels) 
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    
        loss = F.cross_entropy(out, labels)   
        acc = accuracy(out, labels)           
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))
        
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs');

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');


input_size = 3*32*32
output_size = 10
class CIFAR10Model(ImageClassificationBase):
    def __init__(self, input_size = 3*32*32, output_size = 10):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 1024)
        self.linear2 = nn.Linear(1024, 256)
        self.linear3 = nn.Linear(256, 64)
        self.linear4 = nn.Linear(64, output_size)
        
    def forward(self, x):
        output = x.view(x.size(0), -1)
        output = self.linear1(output)
        output = F.relu(output)
        output = self.linear2(output)
        output = F.relu(output)
        output = self.linear3(output)
        output = F.relu(output)
        output = self.linear4(output)
        return output
    

model = CIFAR10Model()
history = [evaluate(model, val_loader)]


history += fit(10, 1e-1, model, train_loader, val_loader)
history += fit(10, 1e-2, model, train_loader, val_loader)
history += fit(10, 1e-3, model, train_loader, val_loader)
history += fit(10, 1e-4, model, train_loader, val_loader)
plot_losses(history)
plot_accuracies(history)
test_data = torchvision.datasets.CIFAR10('./', download=False, train=False, transform=ToTensor())
test_loader = DataLoader(test_data, batch_size*2, num_workers=4)
evaluate(model, test_loader)