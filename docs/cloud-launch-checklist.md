# Guía Rápida de Despliegue en GPU Cloud (`Gate A`)

## 1. ¿RunPod o Vast.ai? ¿Qué GPU elegir?

### Recomendación de Plataforma
- **1ª Opción (Más fiable y sin sorpresas): RunPod (`Secure Cloud` o `Community Cloud`)**
  - **Por qué**: Discos NVMe rápidos, plantillas PyTorch oficiales limpias, SSH directo con 1 clic y almacenamiento persistente en `/workspace`.
- **2ª Opción (Más barato por hora): Vast.ai**
  - **Por qué**: Puedes encontrar GPUs un 30–40 % más baratas, pero son máquinas de particulares (a veces el disco o la red van lentos). Si usas Vast.ai, filtra siempre por **`Reliability > 99%`**, **`Disk Read > 1500 MB/s`** y **`Upload/Download > 500 Mbps`**.

### Recomendación de GPU (Calidad / Precio para Kintsugi)
1. **Mejor Calidad/Precio (Recomendada): `1x NVIDIA L40S (48 GB VRAM)` o `1x RTX 6000 Ada (48 GB VRAM)`**
   - **Precio**: ~\$0,79 – \$1,05 USD/hora en RunPod (~\$0,55 USD/hora en Vast.ai).
   - **Ventaja**: Soporte nativo `bf16-mixed`, 48 GB de sobra para `base_channels=32`, y arquitectura Ada Lovelace rapidísima para `mamba-ssm`.
2. **Opción Económica Rápida: `1x NVIDIA RTX 4090 (24 GB VRAM)`**
   - **Precio**: ~\$0,39 – \$0,59 USD/hora.
   - **Ventaja**: Es muy rápida por vatio/dólar y con parches `16x128x128` + `bf16-mixed` entra holgadamente en 24 GB.
3. **Almacenamiento**: Pide **50 GB de Container Disk / Volume Disk** montado en `/workspace`.

---

## 2. Paso a Paso en la Máquina GPU (3 Comandos)

### Paso A — Clonar tu repo privado e instalar el entorno en el Pod
Conéctate por SSH al Pod de RunPod/Vast.ai y ejecuta:

```bash
cd /workspace
git clone https://github.com/xechug/kintsugi-vesuvius.git
cd kintsugi-vesuvius
pip install --upgrade pip
pip install -r cloud/requirements.txt
```

### Paso B — Subir tus 3 Fragmentos Zarr ya procesados (¡Solo pesan 4,2 GB sin `layers/`!)
En tu Mac tienes `data/public/kaggle2023_fragment{1,2,3}` que ocupa 15 GB en total, pero **10,4 GB son los `.tif` sueltos de `layers/` que ya no hacen falta porque ya están convertidos a `volume.zarr`**.
Los 3 `volume.zarr` + `mask.tif` + `inklabels.tif` + `split.json` pesan **solo 4,2 GB en total**.

Ejecuta este comando **desde la terminal de tu Mac** (cambiando `<IP_DEL_POD>` y `<PUERTO_SSH>` por los que te da RunPod):

```bash
rsync -avzP --exclude='layers' --exclude='*.png' \
  -e "ssh -p <PUERTO_SSH>" \
  /Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge/data/public/ \
  root@<IP_DEL_POD>:/workspace/data/public/
```

### Paso C — Lanzar el Benchmark Controlado (`Gate A`) y Generar la Imagen de Letras Griegas
De vuelta en la terminal SSH del servidor GPU, lanza el entrenamiento dentro de `tmux` para que no se corte si cierras el portátil:

```bash
tmux new -s kintsugi
cd /workspace/kintsugi-vesuvius

# 1. Prueba rápida de 1 semilla (1337) con base_channels=32
SEEDS="1337" BASE_CHANNELS=32 GPU_MODEL="NVIDIA L40S 48GB" ./scripts/run_cloud_gate_a.sh

# 2. Al terminar, generar la lámina visual comparativa de letras griegas en Fragmento 3:
python scripts/render_fragment3_comparison.py \
  --residual-ckpt /workspace/checkpoints/gate_a/residual_unet_seed1337/best.strict.ckpt \
  --axial-ckpt /workspace/checkpoints/gate_a/kintsugi_axial_seed1337/best.strict.ckpt \
  --physical-ckpt /workspace/checkpoints/gate_a/kintsugi_physical_seed1337/best.strict.ckpt \
  --output-png /workspace/fragment3_gate_a_comparison.png
```
