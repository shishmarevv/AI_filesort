import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ImageDataset
from model import classifier
from progressbar import progressbar
from sklearn.model_selection import train_test_split

print("Start...")

batch_size = 32
learning_rate = 0.001
num_epochs = 5

device = torch.device("cpu")

print("Transforming...")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

print("Loading dataset...")

trainset = ImageDataset('train', transform=transform)

_, valset = train_test_split(trainset, test_size=0.2, random_state=21)

train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)

num_classes = len(trainset.classes)
model = classifier(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for image, label in progressbar(train_loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for image, label in progressbar(val_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

torch.save(model.state_dict(), "models/classifier.pth")
