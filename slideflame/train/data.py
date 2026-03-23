# data.py
import json
import math
import torch
from torch.utils.data import DataLoader, Dataset

from data_utils import DataInfo, SharedEpoch
from train_utils import get_cast_dtype


class PathDataset(Dataset):
    """
    Unified dataset for:
      - report-format JSONL: {"file_path": ..., "result": "..."}
      - QA-format JSONL:     {"file_path": ..., "question": "...", "answer": "..."}

    No classification support (organ/diagnosis removed).
    Prompt is always "<image>".
    Slide features may be None (patch-only mode).
    """

    def __init__(
        self,
        jsonl_file,
        tokenizer,
        feature_loader,
        epoch=0,
        max_tokens=312,
        data_format="auto",  # auto | report | qa
    ):
        self.tokenizer = tokenizer
        self.feature_loader = feature_loader
        self.epoch = epoch
        self.max_tokens = max_tokens
        self.data_format = data_format

        self.entries = self._load_entries(jsonl_file)
        if len(self.entries) == 0:
            raise ValueError(f"Empty jsonl: {jsonl_file}")

        # Infer format if requested
        if self.data_format == "auto":
            e0 = self.entries[0]
            if "question" in e0 and "answer" in e0:
                self.data_format = "qa"
            elif "result" in e0:
                self.data_format = "report"
            else:
                raise ValueError(
                    "Could not infer data_format. Expected keys: "
                    "'result' (report) OR 'question'+'answer' (qa)."
                )

    def _load_entries(self, jsonl_file):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def __len__(self):
        return len(self.entries)

    @staticmethod
    def _prompt():
        return "<image>"

    def __getitem__(self, idx):
        entry = self.entries[idx]
        file_path = entry["file_path"]

        prompt = self._prompt()

        if self.data_format == "qa":
            question_text = entry.get("question", "")
            answer_text = entry.get("answer", "")
            context_text = prompt + " " + question_text
            completion_text = f"{answer_text} <|endofchunk|>"

            enc = self.tokenizer(
                context_text,
                completion_text,
                max_length=self.max_tokens,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Mask the entire context (prompt + question) from loss
            context_enc = self.tokenizer(context_text, add_special_tokens=False)
            context_len = len(context_enc["input_ids"])
            labels[:context_len] = -100

            raw_text = f"Question: {question_text} || Answer: {answer_text}"

        elif self.data_format == "report":
            report_text = entry.get("result", "")
            context_text = prompt
            completion_text = f"{report_text} <|endofchunk|>"

            enc = self.tokenizer(
                context_text,
                completion_text,
                max_length=self.max_tokens,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()

            # Mask prompt from loss robustly
            prompt_enc = self.tokenizer(context_text)
            context_len = len(prompt_enc["input_ids"])
            labels[:context_len] = -100

            raw_text = completion_text

        else:
            raise ValueError(f"Unknown data_format={self.data_format}")

        # Mask padding tokens
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        # Mask <image> token just in case
        img_id = self.tokenizer.convert_tokens_to_ids("<image>")
        if img_id is not None and img_id >= 0:
            labels[input_ids == img_id] = -100

        # Load features (slide_features may be None in patch-only mode)
        assert self.feature_loader is not None, f"Feature loader is None for file {file_path}"
        feats = self.feature_loader(file_path)

        patch_features = feats["patch_features"]
        slide_features = feats.get("slide_features", None)

        # Ensure patch has time dim: (T=1, N, D)
        if patch_features.ndim == 2:
            patch_features = patch_features.unsqueeze(0)

        return {
            "file_path": file_path,
            "raw_text": raw_text,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "patch_features": patch_features,
            "slide_features": slide_features,  # may be None
        }


def collate_fn(batch, cast_dtype=None):
    """
    Collate supports slide_features=None (patch-only).
    """
    patch_features = torch.stack([s["patch_features"] for s in batch])

    slide_list = [s.get("slide_features", None) for s in batch]
    has_slide = (slide_list[0] is not None)

    slide_features = None
    if has_slide:
        slide_features = torch.stack(slide_list)

    if cast_dtype is not None:
        patch_features = patch_features.to(dtype=cast_dtype)
        if slide_features is not None:
            slide_features = slide_features.to(dtype=cast_dtype)

    return {
        "file_path": [s["file_path"] for s in batch],
        "raw_text": [s["raw_text"] for s in batch],
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attention_mask": torch.stack([s["attention_mask"] for s in batch]),
        "labels": torch.stack([s["labels"] for s in batch]),
        "images": patch_features,
        "slide_features": slide_features,  # None if patch-only
    }


def build_dataset(args, tokenizer, feature_loader, epoch=0, floor=False):
    shared_epoch = SharedEpoch(epoch=epoch)

    dataset = PathDataset(
        jsonl_file=args.jsonl_file,
        tokenizer=tokenizer,
        feature_loader=feature_loader,
        epoch=epoch,
        max_tokens=args.max_tokens,
        data_format=getattr(args, "data_format", "auto"),
    )
    dataset.epoch = epoch

    sampler = None
    if args.world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=args.world_size,
            rank=args.rank,
            shuffle=True,
        )

    global_batch_size = args.batch_size * args.world_size
    num_samples = args.train_num_samples
    round_fn = math.floor if floor else math.ceil
    num_batches = round_fn(num_samples / global_batch_size)
    num_samples = num_batches * global_batch_size

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.workers,
        drop_last=True,
        prefetch_factor=4,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=lambda b: collate_fn(b, cast_dtype=get_cast_dtype(args.precision)),
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, sampler=sampler, shared_epoch=shared_epoch)


def get_data(args, feature_loader, tokenizer, epoch=0):
    return build_dataset(args, tokenizer, feature_loader, epoch=epoch)