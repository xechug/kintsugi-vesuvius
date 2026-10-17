import os
import torch
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from model import DualStreamVMambaForense
from dataset import PapyriDataset

def generate_prediction_image():
    print("🔍 Iniciando modo LECTURA (Inferencia)...")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Cargar el modelo
    # En un caso real, aquí cargaríamos los pesos entrenados: model.load_state_dict(torch.load('modelo_kintsugi.pth'))
    # Como solo entrenamos 2 parches, usaremos la red inicializada para ver cómo escupe la imagen.
    model = DualStreamVMambaForense(in_channels=1, use_real_mamba=False).to(device)
    
    # Cargar el "cerebro" si ya ha sido entrenado
    model_path = "/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge/modelo_kintsugi.pth"
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("🧠 Pesos del modelo (Kintsugi) cargados. La red usará lo que aprendió.")
    else:
        print("⚠️ No se encontró modelo entrenado, la imagen saldrá aleatoria.")
        
    model.eval() # Modo lectura, apagamos el aprendizaje
    
    # 2. Cargar datos
    zarr_dir = "/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge/sample_data/volume.zarr"
    mask_file = "/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge/sample_data/mask.tif"
    ink_file = "/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge/sample_data/inklabels.tif"
    
    dataset = PapyriDataset(zarr_dir, mask_file, ink_file, patch_size=(32, 64, 64), stride=(32, 32, 32), downsample_factor=2)
    
    print("Buscando un bloque de papiro que SÍ tenga tinta para leer...")
    
    # Buscamos un parche que tenga tinta (Ground Truth > 0)
    encontrado = False
    for idx in range(2000): # Buscamos en los primeros 2000 parches
        vol_tensor, mask_tensor = dataset[idx]
        if mask_tensor.max() > 0:
            print(f"¡Bloque con tinta encontrado en el índice {idx}!")
            encontrado = True
            break
            
    if not encontrado:
        print("No se encontró tinta en los primeros parches, usando el último...")
        
    # Añadimos la dimensión del Batch para que el modelo lo entienda: [1, Channels, Depth, Height, Width]
    vol_batch = vol_tensor.unsqueeze(0).to(device)
    
    # 3. La IA intenta leer la tinta
    with torch.no_grad(): # No gastamos memoria en gradientes
        prediction_2d = model.predict_proba(vol_batch)
    
    # 4. Convertir el tensor matemático a una imagen visible
    pred_image = prediction_2d.squeeze().cpu().numpy()
    real_mask = mask_tensor.squeeze().numpy()
    
    # Normalizar la imagen de predicción entre 0 y 1 para guardarla
    pred_image = (pred_image - pred_image.min()) / (pred_image.max() - pred_image.min() + 1e-8)
    
    # 5. Guardar la imagen comparativa
    output_path = "/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge/prediction_sample.png"
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(pred_image, cmap='magma')
    axes[0].set_title(f"Predicción IA (Max: {pred_image.max():.2f})")
    axes[0].axis('off')
    
    axes[1].imshow(real_mask, cmap='gray')
    axes[1].set_title(f"Tinta Real (Ground Truth)\\n¿Hay tinta aquí?: {'SÍ' if real_mask.max() > 0 else 'NO'}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ¡Papiro leído! Se ha generado una imagen termográfica de la predicción en: {output_path}")

if __name__ == "__main__":
    generate_prediction_image()
