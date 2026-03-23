#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch
import h5py
import pandas as pd

from slideflame import create_model_and_transforms


def parse_args():
    parser = argparse.ArgumentParser("slideflame WSI report generation evaluation")

    # Model / tokenizer
    parser.add_argument("--lang_encoder_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--cross_attn_every_n_layers", type=int, default=1)
    parser.add_argument("--freeze_lm_embeddings", action="store_true")

    # Feature locations
    parser.add_argument("--patch_dir", type=str, required=True,
                        help="Directory with patch-level features (.pt or .h5)")
    parser.add_argument("--patch_ext", type=str, default=".pt",
                        help="Filter extension when scanning patch_dir (default .pt). Use '' to disable filter.")
    parser.add_argument("--slide_dir", type=str, default=None,
                        help="Directory with slide-level .h5 features (optional if --patch_only)")
    parser.add_argument("--slide_ext", type=str, default=".h5",
                        help="Extension for slide feature files (default .h5)")

    # CSV listing cases (optional)
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Optional CSV with a column listing file stems or filenames")
    parser.add_argument("--csv_file_column", type=str, default="File",
                        help="CSV column containing file names / stems")

    # Output
    parser.add_argument("--output_json", type=str, required=True)

    # Prompt and length
    parser.add_argument("--prompt", type=str, default="<image>")
    parser.add_argument("--max_new_tokens", type=int, default=320)
    parser.add_argument("--min_new_tokens", type=int, default=0,
                    help="Force generation of at least this many new tokens before EOS is allowed.")

    # Generation techniques
    parser.add_argument("--decoding", type=str, default="beam",
                        choices=["beam", "greedy", "sample"],
                        help="beam: beam search; greedy: greedy decoding; sample: stochastic sampling")
    parser.add_argument("--num_beams", type=int, default=5)

    # sampling
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--num_return_sequences", type=int, default=1,
                        help="Used mainly for sampling (sample multiple reports per case).")

    # Modes
    parser.add_argument("--patch_only", action="store_true",
                        help="If set, ignore slide_dir and run patch-only mode")
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


# -------------------- Feature loading helpers -------------------- #

def load_patch_feats(path: str) -> torch.Tensor:
    """Return (N, D) float32."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pt":
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict) and "features" in obj:
            feats = obj["features"]
        elif isinstance(obj, torch.Tensor):
            feats = obj
        else:
            raise ValueError(f"Unsupported .pt structure in {path}")
    elif suffix == ".h5":
        with h5py.File(path, "r") as f:
            if "feats" in f:
                arr = f["feats"][:]
            elif "features" in f:
                arr = f["features"][:]
            else:
                keys = list(f.keys())
                if len(keys) != 1:
                    raise ValueError(f"Ambiguous keys in patch h5 file {path}: {keys}")
                arr = f[keys[0]][:]
        feats = torch.tensor(arr, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported patch feature extension: {suffix} for {path}")

    if feats.ndim == 1:
        feats = feats.unsqueeze(0)
    if feats.ndim != 2:
        raise ValueError(f"Expected patch feats [N,D], got {feats.shape} for {path}")
    return feats


def load_slide_feats(path: str) -> torch.Tensor:
    """Return (1, D) float32."""
    with h5py.File(path, "r") as f:
        if "feats" in f:
            arr = f["feats"][:]
        elif "features" in f:
            arr = f["features"][:]
        else:
            keys = list(f.keys())
            if len(keys) != 1:
                raise ValueError(f"Ambiguous keys in slide h5 file {path}: {keys}")
            arr = f[keys[0]][:]

    feats = torch.tensor(arr, dtype=torch.float32)
    if feats.ndim == 1:
        feats = feats.unsqueeze(0)  # (1, D)
    elif feats.ndim == 2 and feats.shape[0] != 1:
        feats = feats.mean(dim=0, keepdim=True)
    elif feats.ndim != 2:
        raise ValueError(f"Expected slide feats [D] or [1,D] or [N,D], got {feats.shape} for {path}")
    return feats

def build_generation_kwargs(args):
    """
    Convert CLI decoding arguments into Hugging Face generate() kwargs.
    """
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
    }

    if args.decoding == "greedy":
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
        })

    elif args.decoding == "beam":
        gen_kwargs.update({
            "do_sample": False,
            "num_beams": args.num_beams,
            "num_return_sequences": 1,
        })

    elif args.decoding == "sample":
        gen_kwargs.update({
            "do_sample": True,
            "num_beams": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "num_return_sequences": args.num_return_sequences,
        })

    else:
        raise ValueError(f"Unsupported decoding mode: {args.decoding}")

    return gen_kwargs

# -------------------- File manifest resolution -------------------- #

def build_manifest_from_csv(csv_path: str, csv_file_column: str):
    df = pd.read_csv(csv_path)
    cols_upper = [c.strip().upper() for c in df.columns]
    if csv_file_column.upper() not in cols_upper:
        raise ValueError(
            f"CSV must contain column '{csv_file_column}' (case-insensitive). "
            f"Found: {df.columns.tolist()}"
        )
    col = df.columns[cols_upper.index(csv_file_column.upper())]
    items = [str(x).strip() for x in df[col].tolist() if str(x).strip()]
    return items


def build_manifest_from_directory(patch_dir: str, patch_ext: str):
    pdir = Path(patch_dir)
    if not pdir.exists():
        raise FileNotFoundError(f"patch_dir not found: {patch_dir}")

    if patch_ext and not patch_ext.startswith("."):
        patch_ext = "." + patch_ext

    if patch_ext:
        files = sorted(pdir.glob(f"*{patch_ext}"))
    else:
        # no filter: take all files
        files = sorted([p for p in pdir.iterdir() if p.is_file()])

    # Return filenames (not full paths) to keep consistent stem logic
    return [f.name for f in files]


# -------------------- Inference -------------------- #

def infer_single(model, tokenizer, device, patch_feats, slide_feats, prompt, generation_kwargs):
    # (N,D) -> (1,1,N,D)
    patch_feats = patch_feats.unsqueeze(0).unsqueeze(0).to(device)

    slide_feats = slide_feats.to(device) if slide_feats is not None else None

    tokenizer.padding_side = "left"
    ids = tokenizer(prompt, return_tensors="pt").to(device)

    model.eval()
    with torch.no_grad():
        # First try: normal generation (no min_new_tokens)
        out = model.generate(
            patch_features=patch_feats,
            slide_features=slide_feats,
            lang_x=ids["input_ids"],
            attention_mask=ids["attention_mask"],
            **generation_kwargs,
        )

    sequences = out.sequences

    # optional: strip prompt prefix if it appears verbatim
    decoded_outputs = []
    for seq in sequences:
        decoded = tokenizer.decode(seq, skip_special_tokens=True).strip()
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):].strip()
        decoded = decoded.replace(" + ", "+").strip()
        decoded_outputs.append(decoded)
    return decoded_outputs


def main():
    args = parse_args()

    device = args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.decoding == "beam" and args.num_beams < 2:
        raise ValueError("--num_beams must be >= 2 for beam decoding.")

    if args.decoding != "sample" and args.num_return_sequences != 1:
        raise ValueError("--num_return_sequences > 1 is only valid for sample decoding.")

    if args.decoding == "sample" and args.temperature <= 0:
        raise ValueError("--temperature must be > 0 for sample decoding.")
    
    generation_kwargs = build_generation_kwargs(args)
    print("Generation kwargs:", generation_kwargs)

    # Build model/tokenizer
    model, tokenizer = create_model_and_transforms(
        lang_encoder_path=args.lang_encoder_path,
        tokenizer_path=args.tokenizer_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
        use_local_files=(args.cache_dir is not None),
        cache_dir=args.cache_dir,
    )
    model.to(device)

    # Load checkpoint (base)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint with {len(missing)} missing and {len(unexpected)} unexpected keys.")

    # Manifest: CSV optional
    if args.csv_path:
        items = build_manifest_from_csv(args.csv_path, args.csv_file_column)
        print(f"Manifest from CSV: {len(items)} items")
    else:
        items = build_manifest_from_directory(args.patch_dir, args.patch_ext)
        print(f"Manifest from directory scan: {len(items)} items")

    patch_dir = Path(args.patch_dir)
    slide_dir = Path(args.slide_dir) if args.slide_dir else None

    results = []
    for item in items:
        # item may be a stem or a filename
        item_path = Path(item)
        patch_name = item_path.name
        if not item_path.suffix:
            # treat as stem
            ext = args.patch_ext if args.patch_ext else ".pt"
            if ext and not ext.startswith("."):
                ext = "." + ext
            patch_name = f"{patch_name}{ext}"

        patch_path = patch_dir / patch_name
        if not patch_path.exists():
            print(f"[WARN] Patch file missing, skipping: {patch_path}")
            continue

        try:
            patch_feats = load_patch_feats(str(patch_path))
        except Exception as e:
            print(f"[ERROR] Patch load failed: {patch_path} :: {e}")
            continue

        slide_feats = None
        if not args.patch_only:
            if slide_dir is None:
                raise ValueError("slide_dir must be provided unless --patch_only is set.")
            slide_name = Path(patch_name).with_suffix(args.slide_ext).name
            slide_path = slide_dir / slide_name
            if not slide_path.exists():
                print(f"[WARN] Slide file missing, skipping: {slide_path}")
                continue
            try:
                slide_feats = load_slide_feats(str(slide_path))
            except Exception as e:
                print(f"[ERROR] Slide load failed: {slide_path} :: {e}")
                continue

        reports = infer_single(
            model=model,
            tokenizer=tokenizer,
            device=device,
            patch_feats=patch_feats,
            slide_feats=slide_feats,
            prompt=args.prompt,
            generation_kwargs=generation_kwargs,
        )

        # define an ID (you can change this if you prefer stem-only IDs)
        image_id = Path(patch_name).with_suffix(".tiff").name

        if args.decoding == "sample" and args.num_return_sequences > 1:
            print(f"ID: {image_id}")
            for i, report in enumerate(reports):
                print(f"[sample {i}] {report}")
            print("-" * 60)
            results.append({
                "id": image_id,
                "reports": reports,
            })
        else:
            report = reports[0] if len(reports) > 0 else ""
            print(f"ID: {image_id}")
            print(report)
            print("-" * 60)
            results.append({
                "id": image_id,
                "report": report,
            })

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} predictions to {out_path}")


if __name__ == "__main__":
    main()