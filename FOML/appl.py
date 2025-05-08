import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import Compose, Resize, ToTensor
import io
import torch.nn as nn
import torch.nn.functional as F
import tempfile
import os


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)  # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = self.accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.sum(preds == labels).float() / labels.size(0)


# -------------------- Blood Type Compatibility --------------------
def get_compatibility(blood_type):
    """Returns who can donate to and who this type can donate to."""
    can_donate_to = {}
    can_receive_from = {}

    if blood_type == 'A+':
        can_donate_to = ['A+', 'AB+']
        can_receive_from = ['A+', 'A-', 'O+', 'O-']
    elif blood_type == 'A-':
        can_donate_to = ['A+', 'A-', 'AB+', 'AB-']
        can_receive_from = ['A-', 'O-']
    elif blood_type == 'B+':
        can_donate_to = ['B+', 'AB+']
        can_receive_from = ['B+', 'B-', 'O+', 'O-']
    elif blood_type == 'B-':
        can_donate_to = ['B+', 'B-', 'AB+', 'AB-']
        can_receive_from = ['B-', 'O-']
    elif blood_type == 'AB+':
        can_donate_to = ['AB+']
        can_receive_from = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
    elif blood_type == 'AB-':
        can_donate_to = ['AB+', 'AB-']
        can_receive_from = ['A-', 'B-', 'AB-', 'O-']
    elif blood_type == 'O+':
        can_donate_to = ['A+', 'B+', 'AB+', 'O+']
        can_receive_from = ['O+', 'O-']
    elif blood_type == 'O-':
        can_donate_to = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
        can_receive_from = ['O-']

    return can_donate_to, can_receive_from


# -------------------- Model Loading --------------------
@st.cache_resource
def load_model(model_path, device, num_classes):
    class FingerprintToBloodGroup(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, num_classes)
            )

        def forward(self, xb):
            return self.network(xb)

    model = FingerprintToBloodGroup()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


# Globals (adjust these)
model_path = 'model.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_classes = ['A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-']
num_classes = len(dataset_classes)
model = load_model(model_path, device, num_classes)


def predict_blood_group_from_path(image_path):
    """Predicts blood group from an image path."""
    transform = Compose([Resize((64, 64)), ToTensor()])
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_class = dataset_classes[predicted.item()]
        return predicted_class
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


# -- Streamlit App --
st.title("Blood Group Prediction from Fingerprint")

uploaded_file = st.file_uploader("Upload a fingerprint image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        image = Image.open(uploaded_file).convert('RGB')
        image.save(tmp_file.name, format="PNG")
        temp_image_path = tmp_file.name

    st.image(image, caption="Uploaded Fingerprint.", use_column_width=True)

    if st.button("Predict Blood Group"):
        predicted_group = predict_blood_group_from_path(temp_image_path)
        if predicted_group:
            st.write(f"## Predicted Blood Group: {predicted_group}")

            # Blood donation compatibility
            can_donate_to, can_receive_from = get_compatibility(predicted_group)
            st.subheader("Blood Donation Compatibility")
            st.write(f"Can donate to: {', '.join(can_donate_to)}")
            st.write(f"Can receive from: {', '.join(can_receive_from)}")

            # Feature explanation (for image data, it's about learned patterns)
            st.subheader("Model Insights")
            st.write("The machine learning model predicts the blood group by analyzing visual patterns and textures present in the fingerprint image. It learns to associate these intricate details with different blood groups during the training process. The specific 'features' the model uses are complex combinations of edges, curves, and other visual elements captured by the convolutional layers of the neural network. The exact numerical values of these learned features are abstract and not easily interpretable as simple, human-understandable measurements.")

            # Feature dictionary (Illustrative - not actual values)
            feature_dict = {
                "Ridge patterns": "Complex curves and whorls",
                "Texture variations": "Fine vs. coarse texture",
                "Edge details": "Sharpness and density of edges",
                "Singular points": "Presence and location of deltas and cores"
            }
            st.write("Feature Dictionary:")
            st.json(feature_dict)

        os.remove(temp_image_path)
