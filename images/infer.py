import torch
from torchvision import transforms
from PIL import Image
from model import classifier

device = torch.device("cpu")

model = classifier(num_classes=5)
model.load_state_dict(torch.load("models/classifier.pth", map_location=device))
model.to(device)
model.eval()

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    return image

def classify_image(image):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

image_path = "data/test/sample.jpg"
predicted_class = classify_image(image_path)
print(f"Предсказанный класс: {predicted_class}")
