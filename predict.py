import os
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from model import DualStreamVMambaForense
from dataset import PapyriDataset

def generate_prediction_image():
    print("🔍 Starting inference mode...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Load model architecture
    model = DualStreamVMambaForense(in_channels=1, use_real_mamba=False).to(device)
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "modelo_kintsugi.pth")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("🧠 Model weights loaded successfully.")
    else:
        print("⚠️ No trained model weights found; using randomized initialization.")
        
    model.eval()
    
    # 2. Load dataset
    zarr_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data/volume.zarr")
    mask_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data/mask.tif")
    ink_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data/inklabels.tif")
    
    dataset = PapyriDataset(zarr_dir, mask_file, ink_file, patch_size=(32, 64, 64), stride=(32, 32, 32), downsample_factor=2)
    
    print("Searching for a papyrus block containing ink...")
    
    found = False
    for idx in range(2000):
        vol_tensor, mask_tensor = dataset[idx]
        if mask_tensor.max() > 0:
            print(f"Ink-bearing patch located at index {idx}!")
            found = True
            break
            
    if not found:
        print("No positive ink patch found in first 2000; falling back to final patch...")
        
    vol_batch = vol_tensor.unsqueeze(0).to(device)
    
    # 3. Model inference
    with torch.no_grad():
        prediction_2d = model.predict_proba(vol_batch)
    
    # 4. Convert output to visible image
    pred_image = prediction_2d.squeeze().cpu().numpy()
    real_mask = mask_tensor.squeeze().numpy()
    
    # Normalize prediction image to [0, 1] range
    pred_image = (pred_image - pred_image.min()) / (pred_image.max() - pred_image.min() + 1e-8)
    
    # 5. Save comparison plot
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_sample.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred_image, cmap='magma')
    axes[0].set_title(f"Model Prediction (Max: {pred_image.max():.2f})")
    axes[0].axis('off')
    
    axes[1].imshow(real_mask, cmap='gray')
    axes[1].set_title(f"Ground Truth Ink\\nContains ink: {'YES' if real_mask.max() > 0 else 'NO'}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Prediction completed. Saved visualization to: {output_path}")

if __name__ == "__main__":
    generate_prediction_image()
