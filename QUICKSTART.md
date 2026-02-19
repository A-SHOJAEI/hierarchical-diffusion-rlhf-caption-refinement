# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check syntax
python3 -m py_compile scripts/*.py
```

## Run Training

```bash
# Full pipeline with default config
python scripts/train.py

# Specific stage only
python scripts/train.py --stage base

# With custom config
python scripts/train.py --config configs/default.yaml

# Ablation study (baseline)
python scripts/train.py --config configs/ablation.yaml
```

## Run Evaluation

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint models/best_rlhf_model.pt \
    --model-type rlhf

# Save results to custom directory
python scripts/evaluate.py \
    --checkpoint models/best_rlhf_model.pt \
    --model-type rlhf \
    --output-dir results/eval_$(date +%Y%m%d)
```

## Run Inference

```bash
# Single image
python scripts/predict.py \
    --checkpoint models/best_rlhf_model.pt \
    --image-url "path/to/image.jpg"

# Multiple images from file
python scripts/predict.py \
    --checkpoint models/best_rlhf_model.pt \
    --image-list images.txt \
    --output predictions.json

# With confidence scores
python scripts/predict.py \
    --checkpoint models/best_rlhf_model.pt \
    --image-url "image.jpg" \
    --show-confidence
```

## Project Structure

```
.
├── configs/
│   ├── default.yaml       # Full hierarchical RLHF config
│   └── ablation.yaml      # Baseline (sparse rewards only)
├── scripts/
│   ├── train.py          # Training pipeline
│   ├── evaluate.py       # Evaluation
│   └── predict.py        # Inference
├── src/hierarchical_diffusion_rlhf_caption_refinement/
│   ├── data/             # Data loading
│   ├── models/           # Models and components
│   ├── training/         # Training pipeline
│   ├── evaluation/       # Metrics
│   └── utils/            # Utilities
└── tests/                # Test suite
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Model architecture
model:
  hidden_dim: 768
  num_layers: 12
  num_heads: 12

# Training hyperparameters
training:
  base_learning_rate: 0.0001
  rlhf_learning_rate: 0.00001
  clip_weight: 0.7          # Dense reward weight
  preference_weight: 0.3    # Sparse reward weight

# Data settings
data:
  train_max_samples: 5000
  val_max_samples: 500
```

## Key Features

1. **Hierarchical Reward Shaping**: Novel adaptive weighting between dense (CLIP) and sparse (preference) rewards
2. **Three-Stage Training**: Base model → Reward model → RLHF fine-tuning
3. **Ablation Study**: Compare full method vs baseline (configs/ablation.yaml)
4. **Comprehensive Evaluation**: CLIP score, CIDEr, BLEU, specificity, human preference win rate

## Troubleshooting

### Import errors
```bash
# Make sure you're in the project root
cd /path/to/hierarchical-diffusion-rlhf-caption-refinement

# Install in development mode
pip install -e .
```

### CUDA out of memory
Reduce batch sizes in config:
```yaml
training:
  base_batch_size: 8    # Default: 16
  rlhf_batch_size: 4    # Default: 8
```

### MLflow errors
Disable MLflow tracking:
```bash
python scripts/train.py --no-mlflow
```

Or in config:
```yaml
logging:
  mlflow_tracking: false
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=hierarchical_diffusion_rlhf_caption_refinement

# Specific test file
pytest tests/test_model.py -v

# Specific test
pytest tests/test_model.py::TestHierarchicalRewardShaper::test_adaptive_weighting -v
```

## Expected Results

Target metrics (after full training):
- CLIP Score: 0.32
- CIDEr: 1.2
- Human Preference Win Rate: 0.68
- Specificity Score: 0.75

Results are saved to `results/evaluation_metrics.json`

## Next Steps

1. Train baseline: `python scripts/train.py --config configs/ablation.yaml`
2. Train full model: `python scripts/train.py --config configs/default.yaml`
3. Compare results in `results/` directory
4. Run ablation analysis to verify hierarchical reward shaping improves performance
