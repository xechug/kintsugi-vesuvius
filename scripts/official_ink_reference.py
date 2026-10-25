"""Run released ScrollPrize models on the unlabeled P800 crop, without scoring training labels."""
from __future__ import annotations

import copy
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

from scripts.controlled_ink_benchmark import sha256, write_json


def extract_logits(output):
    if isinstance(output, dict):
        output = output["ink"]
    if isinstance(output, (list, tuple)):
        output = output[0]
    if output.ndim == 5 and output.shape[2] == 1:
        output = output.squeeze(2)
    if output.ndim != 4 or output.shape[1] != 1:
        raise ValueError(f"Unexpected official prediction shape: {output.shape}")
    return output


@torch.inference_mode()
def infer(model, volume, offset, normalize, device):
    depth = 17
    start = (volume.shape[0]-depth)//2+offset
    slab = volume[start:start+depth]
    size, stride = 128, 96
    height, width = slab.shape[1:]
    ys = sorted(set(range(0,height-size+1,stride)) | {height-size})
    xs = sorted(set(range(0,width-size+1,stride)) | {width-size})
    coords = [(y,x) for y in ys for x in xs]
    axis = np.linspace(-1,1,size,dtype=np.float32)
    weight = np.exp(-2*(axis[:,None]**2+axis[None,:]**2))
    total = np.zeros((height,width),np.float32)
    count = np.zeros_like(total)
    for index in range(0,len(coords),8):
        batch_coords=coords[index:index+8]
        patches=[normalize(np.asarray(slab[:,y:y+size,x:x+size]))[None] for y,x in batch_coords]
        batch=torch.from_numpy(np.stack(patches).astype(np.float32)).to(device)
        with torch.autocast("cuda",dtype=torch.float16):
            logits=extract_logits(model(batch))
        probabilities=logits.float().sigmoid().squeeze(1).cpu().numpy()
        for (y,x),prob in zip(batch_coords,probabilities):
            total[y:y+size,x:x+size]+=prob*weight
            count[y:y+size,x:x+size]+=weight
    return total/np.maximum(count,1e-8)


def main():
    repo=Path(__file__).resolve().parents[1]
    data=repo/"data/controlled_ink_20261025"
    out=repo/"runs/controlled_ink_20261025"
    source=data/"official"
    sys.path[:0]=[str(source/"ink-detection"),str(source/"vesuvius/src")]
    # Import the original model modules without initializing the unrelated CT SDK,
    # which otherwise imports its optional remote-storage and GUI dependencies.
    package = types.ModuleType("vesuvius")
    package.__path__ = [str(source/"vesuvius/src/vesuvius")]
    sys.modules["vesuvius"] = package
    from koine_machines.models.make_model import make_model
    from vesuvius.image_proc.intensity.normalization import normalize_robust

    protocol=json.loads((out/"protocol.json").read_text())
    volume=np.load(data/"p800_volume.npy",mmap_mode="r")
    device=torch.device("cuda")
    torch.set_num_threads(8)
    reference={"source":json.loads((source/"source.json").read_text()),"models":[],
        "evaluation_policy":"Only unlabeled PHerc0800. w039 and w044 are in released models' training data and are not used for independent accuracy comparisons.",
        "normalization":"Official normalize_robust, per 17x128x128 patch; raw sigmoid outputs retained.",
        "native_um":8.64,"depth_offsets":protocol["p800_depth_offsets"]}
    for seed in (42,43):
        checkpoint=source/f"ink9um_seed{seed}_step75000.pth"
        payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        config=copy.deepcopy(payload["config"])
        config["crop_size"]=config.get("crop_size",config["patch_size"])
        model=make_model(config)
        state=next((payload[k] for k in ("model","state_dict","model_state_dict") if isinstance(payload.get(k),dict)),None)
        if state is None:
            raise ValueError("Official checkpoint does not contain model weights")
        state={key.removeprefix("module."):value for key,value in state.items()}
        model.load_state_dict(state,strict=True)
        model=model.to(device).eval()
        started=time.monotonic()
        files=[]
        for offset in protocol["p800_depth_offsets"]:
            probability=infer(model,volume,offset,normalize_robust,device)
            if not np.isfinite(probability).all():
                raise FloatingPointError("Non-finite official probability map")
            name=f"official_seed{seed}_z{offset:+d}.npy"
            np.save(out/name,probability)
            files.append(name)
            print(f"OFFICIAL seed={seed} z={offset:+d} range={probability.min():.4f}..{probability.max():.4f}",flush=True)
        reference["models"].append({"seed":seed,"checkpoint_sha256":sha256(checkpoint),
            "checkpoint_url":f"https://huggingface.co/scrollprize/ink_9um/blob/main/hybrid_3d2d-seed{seed}/step-075000.pth",
            "parameters":sum(p.numel() for p in model.parameters()),"strict_state_load":True,
            "seconds":time.monotonic()-started,"prediction_files":files,"training_config":config})
        write_json(out/"official_reference.json",reference)
        del model,payload,state
        torch.cuda.empty_cache()
    print("OFFICIAL_REFERENCE_COMPLETE",flush=True)


if __name__=="__main__":
    main()
