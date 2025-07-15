from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import torch.nn.functional as F
from skimage.feature import local_binary_pattern
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, static_folder="static", template_folder="templates")  # Ensure correct folders
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///classified_images_4.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
app.app_context().push()

# Uploads directory
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ClassifiedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False)
    image_url = db.Column(db.String(255), nullable=False)

    def __init__(self, category, image_url):
        self.category = category
        self.image_url = image_url

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model architecture
class LBP_CNN(nn.Module):
    def __init__(self, num_classes):
        super(LBP_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(512, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 7)  # Assuming 7 classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.pool(F.relu(self.bn5(self.conv5(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load model
model_path = r"C:\Users\kaupk\OneDrive\Documents\Symbiosis\LBP_model_with_people_without_batik.pth"
checkpoint = torch.load(model_path, map_location=device)
num_classes = checkpoint['num_classes']
model = LBP_CNN(num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# LBP transformation
LBP_RADIUS = 3
LBP_POINTS = 8 * LBP_RADIUS

def apply_lbp(image):
    image_gray = np.array(image.convert("L"))
    lbp = local_binary_pattern(image_gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp = (lbp / lbp.max() * 255).astype(np.uint8)
    lbp_image = cv2.merge([lbp, lbp, lbp])
    return Image.fromarray(lbp_image)

# Image transformations
transform = transforms.Compose([
    transforms.Lambda(apply_lbp),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Class names
class_names = ['Ajrakh','Bagh', 'Bandhani', 'Dabu', 'Ikat','Kalamkari','Leheriya']

# Classification function
def classify_image(image_path, model, class_names):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    return predicted_class

@app.route('/')
def home():
    return render_template("index.html")  # Renders 'index.html' from the templates folder

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    print("Classifying...")

    category = classify_image(filepath, model, class_names)
    print("Done!")
    image_url = f"/static/uploads/{filename}"
    print("Returning image URL:", image_url)
    print(category)

    # Store in database
    new_image = ClassifiedImage(category=category, image_url=image_url)
    db.session.add(new_image)
    db.session.commit()

    return jsonify({"category": category, "image_url": image_url})

@app.route('/get_images', methods=['GET'])
def get_images():
    with app.app_context():
        images = ClassifiedImage.query.all()
        image_data = [{"category": img.category, "image_url": img.image_url} for img in images]
        print(image_data)
    return jsonify(image_data)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    db.create_all()
    app.run(debug=True)
