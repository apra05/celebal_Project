import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from landuse.data import build_transforms, load_image_folder
from landuse.models import load_checkpoint

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        score = output[0, class_idx]
        score.backward()
        
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        # Weighted combination of activations
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam) # ReLU on CAM
        
        cam = cam.cpu().numpy()
        cam = cam - np.min(cam)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        return cam, class_idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", default="runs/resnet18/best.pt")
    parser.add_argument("--out-dir", default="runs/bonus_gradcam")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint.get("classes", [])
    model = load_checkpoint(args.checkpoint, num_classes=len(classes), device=device)
    model.eval()

    # Get target layer for ResNet-18 (last conv layer)
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    dataset = load_image_folder(args.data_dir, train=False)
    transform = build_transforms(train=False)

    # Pick 3 random images
    indices = [0, len(dataset)//2, len(dataset)-1]
    
    for i, idx in enumerate(indices):
        img_path, true_label = dataset.samples[idx]
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        cam, pred_idx = grad_cam.generate(input_tensor, None)
        
        img_np = np.array(img)
        cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(img_np, 0.5, heatmap, 0.5, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Original (True: {classes[true_label]})")
        axes[0].axis('off')
        
        axes[1].imshow(heatmap)
        axes[1].set_title(f"GradCAM Heatmap")
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title(f"Overlay (Pred: {classes[pred_idx]})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(out_dir / f"gradcam_{i+1}.png", dpi=150)
        plt.close()
        
    print(f"Generated 3 GradCAM visualizations in {out_dir}")

if __name__ == "__main__":
    main()
