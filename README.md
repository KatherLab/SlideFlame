# SlideFlame

SlideFlame is a slide-level vision-language model based on the Open-Flamingo architecture, designed for Pathology report generation task. It integrates patch-level and slide-level visual features with language models to generate detailed reports from medical images.

## Features

- **Multimodal Integration**: Combines image patches and whole-slide features with text for comprehensive understanding.
- **Flexible Training**: Supports full fine-tuning, LoRA adaptation, and patch-only modes.
- **Scalable**: Built for distributed training with PyTorch.
- **Report Generation**: Generates pathology reports from patch and optional slide features.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Srividhya-Sainath/SlideFlame.git
   cd SlideFlame
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   For training-specific dependencies:
   ```bash
   pip install -r requirements-training.txt
   ```

3. Set up the environment (optional, using conda):
   ```bash
   conda env create -f environment.yml
   conda activate slideflame
   ```

## Usage

### Model Architecture

The core model (`flamingo.py`) implements a Flamingo-style architecture:
- Language encoder
- Perceiver Resampler for processing visual features
- Cross-attention layers integrated into the language model

### Data Preparation

Data is loaded from JSONL files supporting two formats:
- **Report Format**: `{"file_path": "...", "result": "..."}`
- **QA Format**: `{"file_path": "...", "question": "...", "answer": "..."}`

Use `PathDataset` in `data.py` to load and preprocess data.

### Creating Patch Feature Bags

Training expects precomputed patch feature bags, not raw image patches. A patch bag is one `.pt` file per slide containing a tensor of patch embeddings with shape `[num_patches, 768]`.

The patch feature path passed to `--vision_features` must include `{epoch}`. During epoch `0`, the loader replaces `{epoch}` with `0`; during epoch `1`, it replaces it with `1`, and so on.

Example directory layout:

```text
patch_bags/
  epoch0/
    slide_001.pt
    slide_002.pt
  epoch1/
    slide_001.pt
    slide_002.pt
```

Use the matching template in training:

```bash
--vision_features "patch_bags/epoch{epoch}"
```

The `.pt` filename must match the basename of the `file_path` value in the JSONL file. For example, if a JSONL entry contains:

```json
{"file_path": "/path/to/slides/slide_001.svs", "result": "..."}
```

then the loader looks for:

```text
patch_bags/epoch0/slide_001.pt
patch_bags/epoch1/slide_001.pt
```

Each `.pt` file can be saved as either a tensor directly or as a dictionary with a `features` key:

```python
import torch

# patch_features should be a float tensor with shape [num_patches, 768]
torch.save({"features": patch_features.float()}, "patch_bags/epoch0/slide_001.pt")
```

If the same patch bags should be reused for every training epoch, create the expected `epoch0`, `epoch1`, ... directories by copying or symlinking the same `.pt` files.

### Training

Run multi-GPU training with `torchrun`. `--vision_features` points to patch-level features and must include `{epoch}`.

Patch-only training:

```bash
NUM_GPUS=4

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} slideflame/train/train.py \
  --run_name my_experiment \
  --vision_features "path/to/patch_features_{epoch}" \
  --jsonl_file path/to/data.jsonl \
  --lm_path microsoft/BioGPT \
  --batch_size 64 \
  --num_epochs 100 \
  --patch_only
```

Patch and slide training:

```bash
NUM_GPUS=4

torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} slideflame/train/train.py \
  --run_name my_experiment \
  --vision_features "path/to/patch_features_{epoch}" \
  --slide_features "path/to/slide_features" \
  --jsonl_file path/to/data.jsonl \
  --lm_path microsoft/BioGPT \
  --batch_size 64 \
  --num_epochs 100
```

Key options:
- `--use_lora`: Enable LoRA fine-tuning
- `--patch_only`: Train with patch features only (no slides)
- `--vision_features`: Patch-level feature path or comma-separated paths; must include `{epoch}`
- `--slide_features`: Slide-level feature path or comma-separated paths; required unless `--patch_only` is set
- `--batch_size`: Set batch size
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--report_to_wandb`: Enable Weights & Biases logging

### Report Generation

Generate reports from a trained checkpoint with `slideflame/eval/eval.py`.

Patch-only report generation:

```bash
python slideflame/eval/eval.py \
  --lang_encoder_path microsoft/BioGPT \
  --tokenizer_path microsoft/BioGPT \
  --checkpoint_path path/to/checkpoint.pt \
  --cross_attn_every_n_layers 2 \
  --patch_dir path/to/patch_features \
  --patch_ext .pt \
  --output_json path/to/generated_reports.json \
  --patch_only
```

Patch and slide report generation:

```bash
python slideflame/eval/eval.py \
  --lang_encoder_path microsoft/BioGPT \
  --tokenizer_path microsoft/BioGPT \
  --checkpoint_path path/to/checkpoint.pt \
  --cross_attn_every_n_layers 2 \
  --patch_dir path/to/patch_features \
  --patch_ext .pt \
  --slide_dir path/to/slide_features \
  --slide_ext .h5 \
  --output_json path/to/generated_reports.json
```

Use `--patch_ext .h5` if the patch features are stored as HDF5 files instead of `.pt` files. By default, report generation scans every matching file in `--patch_dir`; use `--csv_path` and `--csv_file_column` to generate reports for a specific list of cases.

## Project Structure

- `slideflame/src/`: Core model implementations
  - `flamingo.py`: Main Flamingo model
  - `helpers.py`: Utility functions
- `slideflame/train/`: Training scripts and data utilities
  - `train.py`: Main training script
  - `data.py`: Dataset classes
- `slideflame/eval/`: Report generation tools

## Contributing

Contributions are welcome! Please submit issues and pull requests.

## Acknowledgements

This repository builds on [OpenFlamingo](https://github.com/mlfoundations/open_flamingo.git). We thank the OpenFlamingo authors for making their work publicly available.

## License

The source code is licensed under the MIT License.
