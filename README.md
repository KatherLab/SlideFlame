# SlideFlame

SlideFlame is a vision-language model based on the Open-Flamingo architecture, designed for Pathology report generation task. It integrates patch-level and slide-level visual features with language models to generate detailed reports from medical images.

## Features

- **Multimodal Integration**: Combines image patches and whole-slide features with text for comprehensive understanding.
- **Flexible Training**: Supports full fine-tuning, LoRA adaptation, and patch-only modes.
- **Scalable**: Built for distributed training with PyTorch.
- **Evaluation**: Includes evaluation scripts for assessing model performance.

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
- Language encoder (e.g., GPT-based model)
- Perceiver Resampler for processing visual features
- Cross-attention layers integrated into the language model

### Data Preparation

Data is loaded from JSONL files supporting two formats:
- **Report Format**: `{"file_path": "...", "result": "..."}`
- **QA Format**: `{"file_path": "...", "question": "...", "answer": "..."}`

Use `PathDataset` in `data.py` to load and preprocess data.

### Training

Run training with command-line arguments:

```bash
python -m slideflame.train.train --run_name my_experiment --vision_features "path/to/features_{epoch}.h5" --jsonl_file path/to/data.jsonl --lm_path facebook/opt-1.3b
```

Key options:
- `--use_lora`: Enable LoRA fine-tuning
- `--patch_only`: Train with patch features only (no slides)
- `--batch_size`: Set batch size
- `--num_epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--report_to_wandb`: Enable Weights & Biases logging

### Evaluation

Evaluate the model using:
```bash
python -m slideflame.eval.eval --model_path path/to/model --data_path path/to/data
```

## Project Structure

- `slideflame/src/`: Core model implementations
  - `flamingo.py`: Main Flamingo model
  - `helpers.py`: Utility functions
- `slideflame/train/`: Training scripts and data utilities
  - `train.py`: Main training script
  - `data.py`: Dataset classes
- `slideflame/eval/`: Evaluation tools

## Contributing

Contributions are welcome! Please submit issues and pull requests.

## License

[Add your license here]