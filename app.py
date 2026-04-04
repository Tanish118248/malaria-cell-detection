# ============================================================
#  MALARIA DETECT AI — Flask Backend Server
#  EST Project UCS321 | AI for Engineers | 4th Semester
# ============================================================
# INSTALL (run once):
#   pip install flask flask-cors torch torchvision pillow
# RUN:
#   python app.py
# ============================================================

import io
import base64
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from torchvision import transforms, models

# ── CONFIG ─────────────────────────────────────────────────
MODEL_PATH = 'malaria_model.pth'   # must be in same folder
DEVICE     = 'cuda' if torch.cuda.is_available() else 'cpu'
CLASSES    = ['Parasitized', 'Uninfected']
IMG_SIZE   = 224

app = Flask(__name__)
CORS(app)   # allows the HTML file to call this server

# ── LOAD MODEL ─────────────────────────────────────────────
def load_model():
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
    print(f"   Device: {DEVICE}")
    return model

model = load_model()

# ── IMAGE TRANSFORM ────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── GRAD-CAM ───────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.target_layer = module
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward(retain_graph=True)
        if self.gradients is None:
            return None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().cpu().numpy()
        if cam.ndim < 2:
            cam = np.expand_dims(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def generate_heatmap_base64(original_img, cam):
    """Overlay Grad-CAM heatmap on original image, return base64 PNG."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    orig = np.array(original_img.resize((224, 224))) / 255.0
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#111116')

    titles = ['Original Cell', 'Grad-CAM Heatmap', 'AI Focus Overlay']
    for ax, title in zip(axes, titles):
        ax.set_facecolor('#111116')
        ax.set_title(title, color='#888899', fontsize=10, pad=8)
        ax.axis('off')

    axes[0].imshow(orig)
    axes[1].imshow(cam_resized, cmap='jet')
    axes[2].imshow(orig)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.45)

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120,
                bbox_inches='tight', facecolor='#111116')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── ROUTES ─────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status'  : 'online',
        'device'  : DEVICE,
        'model'   : MODEL_PATH,
        'classes' : CLASSES,
        'accuracy': '93.95%'
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Load & preprocess image
        img = Image.open(file.stream).convert('RGB')
        input_tensor = transform(img).unsqueeze(0).to(DEVICE)

        # Predict
        with torch.no_grad():
            outputs     = model(input_tensor)
            probs       = torch.softmax(outputs, dim=1)[0]
            pred_idx    = probs.argmax().item()
            pred_label  = CLASSES[pred_idx]
            confidence  = probs[pred_idx].item() * 100
            inf_prob    = probs[0].item() * 100   # Parasitized
            hlt_prob    = probs[1].item() * 100   # Uninfected

        # Grad-CAM
        input_tensor_grad = transform(img).unsqueeze(0).to(DEVICE)
        input_tensor_grad.requires_grad_(True)
        gradcam  = GradCAM(model)
        cam      = gradcam.generate(input_tensor_grad, pred_idx)
        heatmap  = generate_heatmap_base64(img, cam) if cam is not None else None

        # Risk level
        risk = 'High' if (pred_label == 'Parasitized' and confidence > 80) else \
               'Moderate' if pred_label == 'Parasitized' else 'Low'

        return jsonify({
            'prediction'   : pred_label,
            'confidence'   : round(confidence, 2),
            'infected_prob': round(inf_prob, 2),
            'healthy_prob' : round(hlt_prob, 2),
            'risk'         : risk,
            'heatmap_b64'  : heatmap,
            'model_used'   : 'EfficientNet-B0 (trained)',
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── RUN ────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*50)
    print("  🔬 MalariaDetect AI — Flask Server")
    print("="*50)
    print(f"  Model  : {MODEL_PATH}")
    print(f"  Device : {DEVICE}")
    print(f"  URL    : http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)


