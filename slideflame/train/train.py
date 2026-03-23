"""
Unified training script (base training + LoRA fine-tuning + patch-only)

- cls (organ/diagnosis) removed completely
- LoRA checkpoints: saves adapters only (no redundant full weights)
- Patch-only: disables slide features end-to-end
- W&B: only rank0 initializes/logs; non-rank0 hard-disabled
"""

import argparse
import os
import random
import math

import torch
import numpy as np

from data import get_data
from distributed import init_distributed_device, world_info_from_env
from torch.nn.parallel import DistributedDataParallel as DDP

from train_utils import (
    train_one_epoch,
    create_feature_loader,
    filter_state_dict_to_trainable,
)

from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from slideflame import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def maybe_load_peft_adapter(model, adapter_dir: str):
    """
    Load PEFT adapter weights from adapter_dir into an already PEFT-wrapped model.
    Robust across PEFT versions.
    """
    if hasattr(model, "load_adapter"):
        try:
            model.load_adapter(adapter_dir, adapter_name="default", is_trainable=True)
            return model
        except TypeError:
            model.load_adapter(adapter_dir, adapter_name="default")
            return model
        except Exception:
            pass

    from peft import PeftModel
    return PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)


def save_checkpoint_unified(ddp_model, optimizer, lr_scheduler, epoch, args, wandb_run=None):
    """
    Always writes run_name/checkpoint_{epoch}.pt with:
      - epoch, optimizer, scheduler, args

    If --use_lora:
      - saves adapters/config to run_name/ via model.save_pretrained(run_name)
      - DOES NOT store model_state_dict in .pt

    Else:
      - stores ONLY trainable subset (filtered) in model_state_dict inside .pt
    """
    if args.rank != 0:
        return

    os.makedirs(args.run_name, exist_ok=True)
    ckpt_path = os.path.join(args.run_name, f"checkpoint_{epoch}.pt")

    real_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model

    ckpt = {
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        "args": vars(args),
    }

    if args.use_lora:
        # Save adapters/config only (PEFT). No full model weights in .pt.
        real_model.save_pretrained(args.run_name)
    else:
        model_state = real_model.state_dict()
        model_state = filter_state_dict_to_trainable(real_model, model_state)
        ckpt["model_state_dict"] = model_state

    print(f"[rank0] Saving checkpoint to {ckpt_path}")
    torch.save(ckpt, ckpt_path)

    if args.report_to_wandb and args.save_checkpoints_to_wandb and wandb_run is not None:
        # Use the run handle (no global wandb import required)
        wandb_run.save(ckpt_path)

    if args.delete_previous_checkpoint and epoch > 0:
        prev = os.path.join(args.run_name, f"checkpoint_{epoch-1}.pt")
        if os.path.exists(prev):
            os.remove(prev)


def main():
    parser = argparse.ArgumentParser()

    # --- training ---
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--delete_previous_checkpoint", action="store_true")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["constant", "linear", "cosine"])
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=100)

    parser.add_argument("--offline", action="store_true")

    # Optional regularizers you already have
    parser.add_argument("--lambda_gate", type=float, default=0.0)
    parser.add_argument("--lambda_slide_gate", type=float, default=0.0)

    # --- unified data ---
    parser.add_argument("--data_format", type=str, default="auto",
                        choices=["auto", "report", "qa"])

    parser.add_argument("--patch_only", action="store_true",
                        help="Disable slide features entirely (patch-only perceiver).")

    # --- LoRA ---
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_target_modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--custom_base_checkpoint", type=str, default=None,
                        help="Optional full checkpoint to load BEFORE applying LoRA.")

    # --- data paths ---
    parser.add_argument("--vision_features", type=str, required=True,
                        help="Must include '{epoch}'. Can be comma-separated base paths.")
    parser.add_argument("--slide_features", type=str, default="",
                        help="Slide .h5 base path(s). Not required if --patch_only.")
    parser.add_argument("--jsonl_file", type=str, required=True)
    parser.add_argument("--train_num_samples", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=1)

    # --- model ---
    parser.add_argument("--lm_path", type=str, default="facebook/opt-1.3b")
    parser.add_argument("--tokenizer_path", type=str, default="")
    parser.add_argument("--cross_attn_every_n_layers", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=312)
    parser.add_argument("--freeze_lm_embeddings", action="store_true")
    parser.add_argument("--gate_learning_rate", type=float, default=None)
    parser.add_argument("--perceiver", type=str, default=None)

    # --- distributed ---
    parser.add_argument("--dist-url", type=str, default="env://")
    parser.add_argument("--dist-backend", type=str, default="nccl")

    # --- wandb ---
    parser.add_argument("--report_to_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--save_checkpoints_to_wandb", action="store_true")

    args = parser.parse_args()

    if "{epoch}" not in args.vision_features:
        raise ValueError("--vision_features must include '{epoch}'.")

    if (not args.patch_only) and (not args.slide_features):
        raise ValueError("--slide_features is required unless --patch_only is set.")

    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("--save_checkpoints_to_wandb requires --report_to_wandb")

    # env flags
    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        # Only enable this if you truly want STRICT HF offline and models are cached:
        # os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # distributed info
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    # hard-disable wandb on non-rank0 BEFORE any wandb init
    if args.rank != 0:
        os.environ["WANDB_MODE"] = "disabled"

    device_id = init_distributed_device(args)
    random_seed(args.seed, args.rank)

    # model/tokenizer
    model, tokenizer = create_model_and_transforms(
        args.lm_path,
        args.tokenizer_path if args.tokenizer_path else args.lm_path,
        cross_attn_every_n_layers=args.cross_attn_every_n_layers,
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )

    # Optional: load a fully-trained base checkpoint BEFORE LoRA wrapping
    if args.custom_base_checkpoint:
        if args.rank == 0:
            print(f"[rank0] Loading custom base checkpoint: {args.custom_base_checkpoint}")
        ckpt = torch.load(args.custom_base_checkpoint, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)

    # Optional: load perceiver weights
    if args.perceiver is not None:
        def strip_prefix_if_present(sd, prefix="perceiver."):
            return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in sd.items()}

        ckpt = torch.load(args.perceiver, map_location="cpu")
        excluded = ["classifier", "attn_weights"]
        ckpt = {k: v for k, v in ckpt.items() if not any(ex in k.lower() for ex in excluded)}
        ckpt = strip_prefix_if_present(ckpt)
        model.perceiver.load_state_dict(ckpt, strict=False)

        # ensure perceiver trainable
        for block in model.perceiver.layers:
            for submodule in block:
                for p in submodule.parameters():
                    p.requires_grad = True

    # LoRA wrapping
    if args.use_lora:
        from peft import LoraConfig, get_peft_model, TaskType

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        if args.rank == 0 and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    # wandb (rank0 only)
    wandb_run = None
    if args.rank == 0 and args.report_to_wandb:
        import wandb
        try:
            # Do NOT pass start_method (deprecated) and do NOT pass service settings.
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.run_name,
                config=vars(args),
                dir=os.environ.get("WANDB_DIR", None),
            )
        except Exception as e:
            print(f"[rank0] wandb init failed -> disabling wandb. Error: {e}")
            os.environ["WANDB_MODE"] = "disabled"
            wandb_run = None

    # Resume
    resume_from_epoch = 0
    checkpoint = None
    if args.resume_from_checkpoint is not None:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        if args.use_lora:
            adapter_dir = os.path.dirname(args.resume_from_checkpoint)
            model = maybe_load_peft_adapter(model, adapter_dir)
        else:
            msd = checkpoint.get("model_state_dict", {})
            msd = {k.replace("module.", ""): v for k, v in msd.items()}
            model.load_state_dict(msd, strict=False)
        resume_from_epoch = checkpoint["epoch"] + 1

    # device + DDP
    model = model.to(device_id)
    if args.world_size > 1:
        ddp_model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
    else:
        ddp_model = model

    # optimizer param groups (decay/no_decay/gate)
    def get_grouped_params(named_params, base_lr, wd, gate_lr=None, gate_lr_mult=1.0):
        decay, no_decay, gate = [], [], []
        for n, p in named_params:
            if not p.requires_grad:
                continue
            if n.endswith("attn_gate") or n.endswith("ff_gate"):
                gate.append(p)
            elif p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        lr_gate = gate_lr if gate_lr is not None else base_lr * gate_lr_mult
        return [
            {"params": decay, "weight_decay": wd, "lr": base_lr},
            {"params": no_decay, "weight_decay": 0.0, "lr": base_lr},
            {"params": gate, "weight_decay": 0.0, "lr": lr_gate},
        ]

    named_params = ddp_model.named_parameters() if hasattr(ddp_model, "named_parameters") else model.named_parameters()
    params_to_optimize = [
        (n, p) for (n, p) in named_params
        if p.requires_grad and not getattr(p, "exclude_from_optimizer", False)
    ]

    optimizer = torch.optim.AdamW(
        get_grouped_params(params_to_optimize, args.learning_rate, args.weight_decay, gate_lr=args.gate_learning_rate),
        betas=(0.9, 0.999),
    )

    if checkpoint is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            print("[rank0] WARNING: optimizer state mismatch; skipping optimizer load.")

    global_batch_size = args.batch_size * args.world_size
    steps_per_epoch = math.ceil(args.train_num_samples / global_batch_size)
    total_training_steps = steps_per_epoch * args.num_epochs

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
        )
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    if checkpoint is not None and "lr_scheduler_state_dict" in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        except Exception:
            print("[rank0] WARNING: scheduler state mismatch; skipping scheduler load.")

    # train
    if hasattr(ddp_model, "train"):
        ddp_model.train()

    for epoch in range(resume_from_epoch, args.num_epochs):
        train_feature_loader = create_feature_loader(
            patch_path_template=args.vision_features,
            slide_path_template=args.slide_features,
            epoch=epoch,
            augment=False,
            patch_only=args.patch_only,
        )

        train_dataset = get_data(args, train_feature_loader, tokenizer, epoch=epoch)
        train_dataset.set_epoch(epoch)
        train_loader = train_dataset.dataloader

        # IMPORTANT: pass wandb_run as the LAST positional arg (works whether train_one_epoch param is named wandb or wandb_run)
        train_one_epoch(
            args,
            ddp_model,
            epoch,
            train_loader,
            tokenizer,
            optimizer,
            lr_scheduler,
            device_id,
            wandb_run,
        )

        save_checkpoint_unified(ddp_model, optimizer, lr_scheduler, epoch, args, wandb_run=wandb_run)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()