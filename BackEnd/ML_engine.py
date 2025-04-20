import os
import base64
import numpy as np
import cv2
import joblib
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'MLModels')

# Create MLModels directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

CLINICAL_MODEL_PATH = os.path.join(MODELS_DIR, 'best_pcos_clinical_model_1.joblib')
IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, 'PCOS_detection_10_epochs_val_acc_1.0.pth')

# Load clinical model
try:
    clinical_model = joblib.load(CLINICAL_MODEL_PATH)
except Exception as e:
    raise Exception(f"Error loading clinical model: {str(e)}")

# Define image model architecture
class PCOSImageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(3, 12, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 3),
            torch.nn.Conv2d(12, 15, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 3),
            torch.nn.Conv2d(15, 10, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3, 3),
            torch.nn.Flatten(),
            torch.nn.Linear(810, 2),
        )

    def forward(self, x):
        return self.network(x)

# Load image model
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_model = PCOSImageModel()
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    image_model.eval()
except Exception as e:
    raise Exception(f"Error loading image model: {str(e)}")

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def create_shap_plot(feature_contributions):
    plt.clf()
    plt.figure(figsize=(10, 6))
    feature_contributions = feature_contributions.sort_values('Absolute SHAP Value', ascending=False)
    bars = sns.barplot(x="SHAP Value", y="Feature", data=feature_contributions, palette="coolwarm")
    plt.title('Most Influential Features for PCOS Prediction', fontsize=14)
    plt.xlabel('SHAP Value (Impact on Prediction)', labelpad=10)
    plt.ylabel('Feature')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close('all')
    return plot_base64

def predict_clinical(data):
    try:
        df = pd.DataFrame([data])
        prediction = clinical_model.predict(df)[0]
        probabilities = clinical_model.predict_proba(df)[0]
        explainer = shap.TreeExplainer(clinical_model)
        shap_values = explainer.shap_values(df)
        if isinstance(shap_values, list):
            if len(shap_values) > 1:
                shap_values_to_use = shap_values[1]
            else:
                shap_values_to_use = shap_values[0]
        else:
            shap_values_to_use = shap_values
        if shap_values_to_use.ndim > 1 and shap_values_to_use.shape[0] == 1:
            shap_values_flat = shap_values_to_use.flatten()
        else:
            shap_values_flat = shap_values_to_use
        feature_contributions = pd.DataFrame({
            'Feature': df.columns,
            'SHAP Value': shap_values_flat[:len(df.columns)],
            'Input Value': df.values[0]
        })
        feature_contributions['Absolute SHAP Value'] = abs(feature_contributions['SHAP Value'])
        plot_base64 = create_shap_plot(feature_contributions)
        feature_contributions = feature_contributions.sort_values('Absolute SHAP Value', ascending=False)
        feature_importance = []
        for _, row in feature_contributions.iterrows():
            feature_importance.append({
                'Feature': row['Feature'],
                'SHAP Value': float(row['SHAP Value']),
                'Absolute SHAP Value': float(row['Absolute SHAP Value']),
                'Input Value': float(row['Input Value'])
            })
        return {
            'prediction': 'NO PCOS' if prediction == 0 else 'PCOS',
            'probabilities': {'NO PCOS': float(probabilities[0]), 'PCOS': float(probabilities[1])},
            'shap_plot': {'NO PCOS': {}} if prediction == 0 else {'PCOS': plot_base64},
            'feature_importance': feature_importance
        }
    except Exception as e:
        print(f"Error details: {str(e)}")
        raise Exception(f"Error in clinical prediction: {str(e)}")

def predict_image(image_file):
    try:
        temp_dir = os.path.join(BASE_DIR, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, 'temp.jpg')
        image_file.save(temp_path)
        img = Image.open(temp_path)
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = image_model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            prediction = output.argmax(dim=1).item()
        def detect_ovarian_cysts(img_tensor):
            if len(img_tensor.shape) == 4:
                img_tensor = img_tensor.squeeze(0)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.shape[2] == 3 else img_np[:,:,0]
            img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_clahe = clahe.apply(img_norm)
            img_inv = 255 - img_clahe
            binary = cv2.adaptiveThreshold(img_inv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 71, -9)
            opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=3)
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cyst_mask = np.zeros_like(img_gray)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 50 or area > img_gray.shape[0] * img_gray.shape[1] * 0.25:
                    continue
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity < 0.4:
                    continue
                if len(contour) < 5:
                    continue
                try:
                    (x, y), (major, minor), angle = cv2.fitEllipse(contour)
                    if minor == 0:
                        continue
                    eccentricity = np.sqrt(1 - (minor/major)**2)
                    if eccentricity > 0.85:
                        continue
                    mask = np.zeros_like(img_gray)
                    cv2.drawContours(mask, [contour], 0, 1, -1)
                    mean_intensity = np.mean(img_gray[mask == 1])
                    if mean_intensity < 0.5:
                        cv2.drawContours(cyst_mask, [contour], 0, 1, -1)
                except:
                    continue
            cyst_mask = cv2.morphologyEx(cyst_mask.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            return cyst_mask
        original_img = img_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        original_img = (original_img - original_img.min()) / (original_img.max() - original_img.min())
        cyst_mask = detect_ovarian_cysts(img_tensor)
        dilated_mask = cv2.dilate(cyst_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        cyst_boundaries = cv2.bitwise_xor(dilated_mask, cyst_mask)
        focus_context = original_img.copy()
        alpha = 0.7
        for i in range(3):
            if i == 0:
                focus_context[:,:,i] = np.where(cyst_mask == 1, alpha * 1.0 + (1-alpha) * focus_context[:,:,i], focus_context[:,:,i])
            else:
                focus_context[:,:,i] = np.where(cyst_mask == 1, alpha * 0.0 + (1-alpha) * focus_context[:,:,i], focus_context[:,:,i])
        for i in range(3):
            if i < 2:
                focus_context[:,:,i] = np.where(cyst_boundaries == 1, 1.0, focus_context[:,:,i])
            else:
                focus_context[:,:,i] = np.where(cyst_boundaries == 1, 0.0, focus_context[:,:,i])
                
        plt.figure(figsize=(20, 6))
        plt.subplot(131)
        plt.imshow(original_img)
        plt.title('Original Ultrasound', fontsize=14)
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(cyst_mask, cmap='gray')
        plt.title('Detected Cysts', fontsize=14)
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(focus_context)
        plt.title('Final Cyst Detection', fontsize=14)
        plt.axis('off')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='jpg', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        combined_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        os.remove(temp_path)
        return {
            'prediction': 'NO PCOS' if prediction == 1 else 'PCOS',
            'probabilities': {'NO PCOS': float(probabilities[1])} if prediction == 1 else {'PCOS': float(probabilities[0])},
            'visualization': {'NO PCOS': {}} if prediction == 1 else {'PCOS': combined_image_base64}
        }
    except Exception as e:
        raise Exception(f"Error in image prediction: {str(e)}")