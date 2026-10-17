import os
import zarr
import tifffile
import numpy as np
import matplotlib.pyplot as plt

def main():
    base_dir = "/Users/jsmcore/.gemini/antigravity/scratch/vesuvius-challenge/sample_data"
    
    # 1. Load the mask
    mask_path = os.path.join(base_dir, "mask.tif")
    if os.path.exists(mask_path):
        mask = tifffile.imread(mask_path)
        print(f"Loaded mask: shape {mask.shape}, dtype {mask.dtype}, min {mask.min()}, max {mask.max()}")
    else:
        print(f"Mask not found at {mask_path}")
        mask = None
        
    # 2. Load the Zarr volume
    zarr_path = os.path.join(base_dir, "volume.zarr")
    if os.path.exists(zarr_path):
        # The OME-Zarr structure usually puts the highest resolution at "0"
        z = zarr.open(zarr_path, mode='r')
        if '0' in z:
            vol = z['0']
            print(f"Loaded Zarr volume '0': shape {vol.shape}, chunks {vol.chunks}, dtype {vol.dtype}")
        else:
            vol = z
            print(f"Loaded Zarr volume root: shape {vol.shape}, chunks {vol.chunks}, dtype {vol.dtype}")
            
        # Get a middle slice
        mid_z = vol.shape[0] // 2
        
        # Let's handle 3D, 4D or 5D just in case
        if len(vol.shape) == 3:
            slice_img = vol[mid_z, :, :]
        elif len(vol.shape) == 5:
            # Usually (t, c, z, y, x)
            slice_img = vol[0, 0, mid_z, :, :]
        else:
            slice_img = vol[mid_z, ...]
            
        print(f"Extracted middle slice at z={mid_z}: min {slice_img.min()}, max {slice_img.max()}")
    else:
        print(f"Zarr volume not found at {zarr_path}")
        slice_img = None
        vol = None
        
    # 3. Visualize
    if mask is not None and slice_img is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Sometimes images need percentile clipping to look good
        vmax = np.percentile(slice_img, 99.5)
        vmin = np.percentile(slice_img, 0.5)
        
        axes[0].imshow(slice_img, cmap='gray', vmin=vmin, vmax=vmax)
        axes[0].set_title(f"CT Slice (z={mid_z})")
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Ink Mask (Ground Truth)")
        axes[1].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(base_dir, "visualization.png")
        plt.savefig(output_path, dpi=150)
        print(f"Visualization saved to {output_path}")
        
        # Optional: Show the plot if running interactively
        # plt.show()

if __name__ == "__main__":
    main()
