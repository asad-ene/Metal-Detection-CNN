import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models
from PIL import Image
import os
import csv
import json

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

print("Classes found (this exact order matters for the app):", image.classes)


# CNN
class DefectModel(nn.Module):
    def __init__(self, num_classes):
        super(DefectModel, self).__init__()
        self.model = models.resnet18(weights='IMAGENET1K_V1')  # pretrained
        self.model.fc = nn.Linear(512, num_classes)  # replace final layer only

    def forward(self, x):
        x = self.model(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DefectModel(num_classes=len(image.classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
best_val_acc = 0.0
SAVE_PATH = 'D:/Python and ML/Metal defects/defect_model.pth'

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

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total
    print(f'Epoch {epoch+1}  Loss: {running_loss/len(dataloader):.4f}  Val Acc: {val_acc:.2f}%')

    # Save whenever validation accuracy improves, so you end up with the best checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'classes': image.classes,
            'num_classes': len(image.classes),
            'val_acc': val_acc,
        }, SAVE_PATH)
        print(f'  -> Saved new best model ({val_acc:.2f}% val acc) to {SAVE_PATH}')

print(f'\nTraining complete. Best val acc: {best_val_acc:.2f}%')
print(f'Model saved at: {SAVE_PATH}')
print(f'Classes (order used by the model): {image.classes}')

# Also dump classes to a small JSON file as a human-readable backup/reference
with open('D:/Python and ML/Metal defects/classes.json', 'w') as f:
    json.dump(image.classes, f, indent=2)


# ---------------------------------------------------------------------------
# logging setup (kept from your original script, for local/manual testing)
# ---------------------------------------------------------------------------
LOG_FILE = 'D:/Python and ML/Metal defects/predictions_log.csv'
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['predicted_class', 'confidence', 'status'])


def log_prediction(image_path, predicted_class, confidence, status):
    row = [image_path, predicted_class, f'{confidence:.4f}', status]
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except PermissionError:
        # The main log is likely open in Excel (or another program) and
        # locked for writing. Don't crash the run — write to a separate
        # backup file instead, and tell the user what happened.
        backup_path = LOG_FILE.replace('.csv', '_locked_backup.csv')
        file_is_new = not os.path.exists(backup_path)
        with open(backup_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if file_is_new:
                writer.writerow([ 'image_path', 'predicted_class', 'confidence', 'status'])
            writer.writerow(row)
        print(
            f'WARNING: {LOG_FILE} is locked (probably open in Excel). '
            f'Close it before the next run. This result was saved to '
            f'{backup_path} instead so nothing was lost.'
        )


def predict_image(image_path,model, dataset, threshold=0.85):
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