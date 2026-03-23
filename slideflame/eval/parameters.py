#!/usr/bin/env python3
"""
Load SlideFlame model + checkpoint and print:
- total parameters
- trainable parameters
- fp32 VRAM needed to keep weights on GPU ("sit on GPU")
No inference / generation is performed.
"""

import os
import torch
from slideflame import create_model_and_transforms

# -----------------------------
# Config (keep your originals)
# -----------------------------
cache_dir = "/data/horse/ws/srsa552c-slideflame/srsa552c-WSIFlamingo-1768996663/srsa552c-CONCHLLAVA-1761087610/srsa552c-CONCHLLAVA-1760914812"

checkpoint_path = (
    "/data/horse/ws/srsa552c-slideflame/srsa552c-WSIFlamingo-1768996663/srsa552c-CONCHLLAVA-1761087610/srsa552c-CONCHLLAVA-1760914812/model/PatchVsSlide/SFBase-B64-GPT4-BioGPT-XATTN2-ATTN1-PatchSlide-SingleText/checkpoint_99.pt"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# -----------------------------
# Helpers
# -----------------------------
def model_size_report(model, device: str):
    """
    Forces fp32 and moves model to device, then prints:
    - total params
    - trainable params
    - fp32 bytes for params+buffers
    - CUDA allocated/reserved (if on GPU)
    """
    # Force fp32 for "VRAM in float32"
    model = model.to(dtype=torch.float32, device=device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    params_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = params_bytes + buffers_bytes

    print("\n==============================")
    print(" Model size diagnostics (fp32)")
    print("==============================")
    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Parameter bytes      : {params_bytes / (1024**3):.3f} GiB")
    print(f"Buffer bytes         : {buffers_bytes / (1024**3):.3f} GiB")
    print(f"Total (weights only) : {total_bytes / (1024**3):.3f} GiB")
    print(f"Model dtype          : {next(model.parameters()).dtype}")

    if device == "cuda":
        torch.cuda.synchronize()
        print(f"CUDA allocated       : {torch.cuda.memory_allocated() / (1024**3):.3f} GiB")
        print(f"CUDA reserved        : {torch.cuda.memory_reserved() / (1024**3):.3f} GiB")

    print("==============================\n")

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "weights_only_gib_fp32": total_bytes / (1024**3),
    }

# -----------------------------
# Build model (same args as you)
# -----------------------------
model, tokenizer = create_model_and_transforms(
    lang_encoder_path="microsoft/BioGPT",
    tokenizer_path="microsoft/BioGPT",
    cross_attn_every_n_layers=2,
    freeze_lm_embeddings=False,
    use_local_files=False,
    cache_dir=cache_dir,
)
tokenizer.padding_side = "left"

# -----------------------------
# Load checkpoint SAFELY on CPU
# (prevents inflating GPU memory)
# -----------------------------
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, map_location="cpu")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    print('Found key "model_state_dict" in checkpoint')
    checkpoint = checkpoint["model_state_dict"]

# Strip DDP prefix if present
if isinstance(checkpoint, dict):
    checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}

missing, unexpected = model.load_state_dict(checkpoint, strict=False)
print(f"Loaded checkpoint. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")

# -----------------------------
# Optional: print gate values
# (no inference; just reads params)
# -----------------------------
gate_values = []
if hasattr(model, "lang_encoder") and hasattr(model.lang_encoder, "gated_cross_attn_layers"):
    for layer in model.lang_encoder.gated_cross_attn_layers:
        if layer is not None and hasattr(layer, "attn_gate"):
            gate_values.append(round(layer.attn_gate.item(), 4))
        else:
            gate_values.append(None)
    print("Gate values:", gate_values)

# -----------------------------
# Diagnostics (this is the goal)
# -----------------------------
report = model_size_report(model, device=device)