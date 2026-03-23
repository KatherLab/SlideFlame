import time
from contextlib import suppress
import os

import torch
from tqdm import tqdm
import h5py


def create_feature_loader(
    patch_path_template: str,
    slide_path_template: str,
    epoch: int = 0,
    augment: bool = False,
    patch_only: bool = False,
):
    """
    Loads patch-level .pt features and (optionally) slide-level .h5 features.

    patch_path_template: comma-separated list of base dirs containing per-slide .pt files
                         (supports {epoch} formatting)
    slide_path_template: comma-separated list of base dirs containing per-slide .h5 files
                         (supports {epoch} formatting); ignored if patch_only=True
    """
    patch_path_template = patch_path_template.split(",")
    slide_path_template = slide_path_template.split(",") if slide_path_template else []

    def feature_loader(file_path: str):
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        patch_filename = base_filename + ".pt"
        slide_filename = base_filename + ".h5"

        # -------------------------
        # Patch features (.pt)
        # -------------------------
        patch_pt_file = None
        for base_path in patch_path_template:
            candidate_path = os.path.join(base_path.format(epoch=epoch), patch_filename)
            if os.path.exists(candidate_path):
                patch_pt_file = candidate_path
                break
        if patch_pt_file is None:
            raise FileNotFoundError(f"Patch feature file not found in any base path: {patch_filename}")

        try:
            patch_data = torch.load(patch_pt_file, map_location="cpu", weights_only=True)
        except (EOFError, RuntimeError) as e:
            print(f"[ERROR: torch.load failed for patch] Epoch: {epoch}, File: {patch_pt_file}, Error: {e}")
            raise

        if isinstance(patch_data, dict) and "features" in patch_data:
            patch_feats = patch_data["features"]
        elif isinstance(patch_data, torch.Tensor):
            patch_feats = patch_data
        else:
            raise ValueError(f"Invalid .pt structure for patch file: {patch_pt_file}")

        if patch_feats.ndim != 2 or patch_feats.shape[1] != 768:
            raise ValueError(f"Expected patch shape [N, 768] in file {patch_pt_file}, but got {patch_feats.shape}")

        if augment:
            idx = torch.randperm(patch_feats.size(0))
            patch_feats = patch_feats[idx]

        patch_feats = patch_feats.to(dtype=torch.float32)

        # -------------------------
        # Slide features (.h5) optional
        # -------------------------
        slide_feats = None
        if not patch_only:
            slide_h5_file = None
            for base_path in slide_path_template:
                try:
                    formatted_base_path = base_path.format(epoch=epoch)
                except KeyError:
                    formatted_base_path = base_path

                candidate_path = os.path.join(formatted_base_path, slide_filename)
                if os.path.exists(candidate_path):
                    slide_h5_file = candidate_path
                    break

            if slide_h5_file is None:
                raise FileNotFoundError(f"Slide feature file not found in any base path: {slide_filename}")

            try:
                with h5py.File(slide_h5_file, "r") as f:
                    # common key name used earlier: "feats"
                    if "feats" in f:
                        slide_feats_np = f["feats"][:]
                    else:
                        # fallback to first dataset
                        first_key = list(f.keys())[0]
                        slide_feats_np = f[first_key][:]
                slide_feats = torch.from_numpy(slide_feats_np)
            except Exception as e:
                print(f"[ERROR: h5py load failed for slide] File: {slide_h5_file}, Error: {e}")
                raise

            if slide_feats.ndim == 1:
                slide_feats = slide_feats.unsqueeze(0)
            if slide_feats.ndim != 2 or slide_feats.shape[0] != 1:
                raise ValueError(
                    f"Expected slide shape [1, D] in file {slide_h5_file}, but got {slide_feats.shape}"
                )

            slide_feats = slide_feats.to(dtype=torch.float32).squeeze(0)

        return {
            "file_path": file_path,
            "patch_features": patch_feats,
            "slide_features": slide_feats,  # None if patch_only=True
        }

    return feature_loader


def get_cast_dtype(precision: str):
    if precision == "bf16" or precision == "bfloat16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return None


def get_mp_policy_dtype(precision: str):
    if "bfloat16" in precision or "bf16" in precision:
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    return torch.float32


def get_autocast(precision, cache_enabled=True):
    """
    Return a context manager factory for autocasting.
    Use torch.amp.autocast (new API) when available to avoid FutureWarning.
    The caller should use: `with autocast(): ...` where autocast = get_autocast(...)
    """
    # Prefer torch.amp.autocast if available (PyTorch >= 2.x)
    try:
        amp_autocast = torch.amp.autocast
    except AttributeError:
        amp_autocast = None

    if precision == "amp":
        if amp_autocast is not None:
            return lambda: amp_autocast(device_type="cuda", enabled=True)
        else:
            # backward compatibility
            return lambda: torch.cuda.amp.autocast(cache_enabled=cache_enabled)
    if precision in ["amp_bfloat16", "amp_bf16", "bf16", "bfloat16"]:
        if amp_autocast is not None:
            return lambda: amp_autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)
        else:
            return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16, cache_enabled=cache_enabled)
    if precision == "fp16":
        if amp_autocast is not None:
            return lambda: amp_autocast(device_type="cuda", dtype=torch.float16, enabled=True)
        else:
            return lambda: torch.cuda.amp.autocast(dtype=torch.float16, cache_enabled=cache_enabled)
    return suppress


def train_one_epoch(
    args,
    model,
    epoch,
    train_loader,
    tokenizer,
    optimizer,
    lr_scheduler,
    device_id,
    wandb_run=None,  # now expects a wandb run object or None
):
    """
    Training loop for report generation (no classification heads).
    Supports patch-only mode by allowing batch["slide_features"] to be None.
    """
    num_batches_per_epoch = train_loader.num_batches
    if args.rank == 0:
        print("Number of batches in training dataset:", num_batches_per_epoch)

    autocast = get_autocast(args.precision)
    cast_dtype = get_cast_dtype(args.precision)

    # Ensure correct reference to underlying model (DDP vs single)
    real_model = model.module if hasattr(model, "module") else model
    model.train()

    step_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for local_step, batch in tqdm(
        enumerate(train_loader),
        disable=args.rank != 0,
        total=num_batches_per_epoch,
    ):
        global_step = epoch * num_batches_per_epoch + local_step

        # data time
        data_time_m.update(time.time() - end)

        images = batch["images"].to(device_id, dtype=cast_dtype, non_blocking=True)

        slide_features = batch.get("slide_features", None)
        if slide_features is not None:
            slide_features = slide_features.to(device_id, dtype=cast_dtype, non_blocking=True)

        input_ids = batch["input_ids"].to(device_id, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device_id, non_blocking=True)
        labels = batch["labels"].to(device_id, non_blocking=True)

        if args.rank == 0 and epoch == 0 and local_step == 0:
            img_id = tokenizer.convert_tokens_to_ids("<image>")
            print("[DEBUG] tokenizer <image> id:", img_id)
            print("[DEBUG] <image> count in input_ids:", (input_ids == img_id).sum().item())
            print("[DEBUG] first sample decoded:", tokenizer.decode(input_ids[0]))

            print("\n[DEBUG] Perceiver grad diagnostics:")
            any_found = False
            for name, param in model.named_parameters():
                if "perceiver" in name and param.requires_grad:
                    any_found = True
                    if param.grad is None:
                        print(f"perceiver grad NONE: {name}")
                    else:
                        gsum = param.grad.abs().sum().item()
                        gmax = param.grad.abs().max().item()
                        print(f"perceiver grad sum={gsum:.3e} max={gmax:.3e}: {name}")
            if not any_found:
                print("NO perceiver params found with requires_grad=True")

        with autocast():
            output = model(
                patch_features=images,
                slide_features=slide_features,  # can be None in patch-only
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # HF CausalLM output compatibility
            loss = output["loss"] if isinstance(output, dict) else output.loss

            # Gate regularization (cross-attn gates)
            if getattr(args, "lambda_gate", 0.0) > 0:
                all_attn_gates = [
                    layer.attn_gate
                    for layer in real_model.lang_encoder.gated_cross_attn_layers
                    if layer is not None
                ]
                if len(all_attn_gates) > 0:
                    gate_reg_loss = -torch.stack([g.tanh() for g in all_attn_gates]).mean()
                    loss = loss + args.lambda_gate * gate_reg_loss

            # Slide gate reg only if slide conditioning is active
            if (
                getattr(args, "lambda_slide_gate", 0.0) > 0
                and slide_features is not None
                and hasattr(real_model.perceiver, "slide_gate")
            ):
                slide_gate_value = real_model.perceiver.slide_gate
                loss = loss + args.lambda_slide_gate * slide_gate_value.pow(2)

            if torch.isnan(loss):
                if args.rank == 0:
                    print("loss is nan, skipping this batch")
                    print("input_ids:", tokenizer.batch_decode(input_ids))
                optimizer.zero_grad(set_to_none=True)
                end = time.time()
                continue

        (loss / args.gradient_accumulation_steps).backward()

        if args.rank == 0 and epoch == 0 and local_step == 0:
            print("\nTrainable layers with non-zero gradients:")
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and param.grad.abs().sum() > 0:
                    print(f"✓ {name}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        do_step = (((local_step + 1) % args.gradient_accumulation_steps) == 0) or (
            local_step == num_batches_per_epoch - 1
        )

        if do_step:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            step_time_m.update(time.time() - end)
            end = time.time()

            if args.rank == 0 and args.report_to_wandb and (wandb_run is not None):
                samples_per_second_per_gpu = (
                    args.gradient_accumulation_steps * args.batch_size / step_time_m.val
                )
                ar_loss_val = (output["loss"].item() if isinstance(output, dict) else output.loss.item())

                wandb_log = {
                    "data_time": data_time_m.avg,
                    "step_time": step_time_m.avg,
                    "epoch": epoch,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr_decay": optimizer.param_groups[0]["lr"],
                    "lr_no_decay": optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else optimizer.param_groups[0]["lr"],
                    "lr_gate": optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else optimizer.param_groups[0]["lr"],
                    "global_step": global_step,
                    "ar_loss": ar_loss_val,
                    "total_loss": loss.item(),
                }

                if hasattr(real_model.perceiver, "slide_gate"):
                    wandb_log["slide_gate_value"] = real_model.perceiver.slide_gate.item()

                # log via the provided wandb_run object
                try:
                    wandb_run.log(wandb_log, commit=True)
                except Exception as e:
                    # avoid training crash for transient wandb issues on rank0
                    print(f"[WARN] wandb logging failed: {e}")

                step_time_m.reset()
                data_time_m.reset()

            if args.rank == 0 and ((local_step + 1) % args.logging_steps == 0):
                print(
                    f"Step {local_step+1}/{num_batches_per_epoch} of epoch {epoch+1}/{args.num_epochs} complete. "
                    f"Loss: {loss.item():.3f}"
                )
        else:
            end = time.time()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def filter_state_dict_to_trainable(model, state_dict):
    """
    Remove non-trainable parameters from model state dict.
    Exception: Embeddings will not be removed, even if frozen.
    This keeps special-token embeddings consistent across runs.
    """
    for name, p in model.named_parameters():
        if "embed" in name or isinstance(p, torch.nn.Embedding):
            continue
        if not p.requires_grad:
            name = name.replace("._checkpoint_wrapped_module", "")
            if name in state_dict:
                del state_dict[name]
            else:
                print(f"WARNING: filtering but {name} not in state_dict")

    # Remove duplicated / unwanted keys
    to_delete = [
        n
        for n in list(state_dict.keys())
        if ("lang_encoder.old_decoder_blocks" in n)
        or ("lang_encoder.gated_cross_attn_layers" in n)
        or ("vision_encoder" in n)
    ]
    for name in to_delete:
        if name in state_dict:
            del state_dict[name]
    return state_dict