import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image
import os
import csv
from datetime import datetime

# transform
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# dataset
image = datasets.ImageFolder('D:/Python and ML/Metal defects/archive/NEU-DET/train/images', transform=train_transform) 
# Create dataloader
dataloader = DataLoader(image, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder('D:/Python and ML/Metal defects/archive/NEU-DET/validation/images', transform=valid_transform)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#CNN
class DefectModel(nn.Module):
    def __init__(self, num_classes):
        super(DefectModel, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')  # pretrained
        self.model.fc = nn.Linear(512, num_classes)  # replace final layer only

    def _get_flattened_size(self, in_channels):
        dummy = torch.zeros(1, in_channels, 32,32)  # fake image
        x = self.pool(torch.relu(self.conv1(dummy)))
        x = self.pool(torch.relu(self.conv2(x)))
        return x.view(1, -1).shape[1]
    
    def forward(self, x):
        x = self.model(x)
        return x
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DefectModel(num_classes=len(image.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
for epoch in range(20):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader):.4f}')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f'Epoch {epoch+1}  Loss: {running_loss/len(dataloader):.4f}  Train Acc: {100*correct/total:.2f}%')

# logging setup
LOG_FILE = 'D:/Python and ML/Metal defects/predictions_log.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([ 'image_path', 'predicted_class', 'confidence', 'status'])

def log_prediction(image_path, predicted_class, confidence, status):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            image_path,
            predicted_class,
            f'{confidence:.4f}',
            status
        ])

# predict
def predict_image(image_path, model, dataset, threshold=0.85):
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)
        confidence = confidence.item()
        predicted_class = dataset.classes[predicted_idx.item()]

    if confidence < threshold:
        status = 'UNCERTAIN'
        print(f'Uncertain prediction — flagged for human review (confidence: {confidence:.2f})')
    else:
        status = 'OK'
        print(f'Predicted class: {predicted_class}  (confidence: {confidence:.2f})')

    log_prediction(image_path, predicted_class, confidence, status)

predict_image('D:/Python and ML/Metal defects/archive/NEU-DET/inclusion_254.jpg', model, image)
