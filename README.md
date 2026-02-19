# Hierarchical Diffusion RLHF Caption Refinement

A two-stage generative system combining diffusion models with reinforcement learning from human feedback (RLHF) for image caption generation. The novel contribution is a hierarchical reward shaping mechanism that combines dense CLIP-based rewards with sparse human preferences using adaptive weighting based on generation confidence.

## Key Features

- Diffusion-based language model for initial caption generation
- RLHF refinement with hierarchical reward shaping
- Adaptive weighting between dense (token-level) and sparse (sequence-level) rewards
- Confidence-based reward balancing for fine-grained quality control

## Installation

```bash
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

Train the full pipeline with default configuration:

```bash
python scripts/train.py
```

Train with custom configuration:

```bash
python scripts/train.py --config configs/default.yaml
```

Train specific stages:

```bash
# Base model only
python scripts/train.py --stage base

# Reward model only
python scripts/train.py --stage reward

# RLHF fine-tuning only
python scripts/train.py --stage rlhf
```

### Ablation Study

Compare hierarchical reward shaping against baseline:

```bash
# Train with hierarchical rewards (default)
python scripts/train.py --config configs/default.yaml

# Train baseline (sparse rewards only)
python scripts/train.py --config configs/ablation.yaml
```

### Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py --checkpoint models/best_rlhf_model.pt --model-type rlhf
```

### Inference

Generate captions for new images:

```bash
# Single image
python scripts/predict.py --checkpoint models/best_rlhf_model.pt --image-url image.jpg

# Multiple images
python scripts/predict.py --checkpoint models/best_rlhf_model.pt --image-list images.txt --output predictions.json
```

## Project Structure

```
hierarchical-diffusion-rlhf-caption-refinement/
├── src/hierarchical_diffusion_rlhf_caption_refinement/
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model implementations and custom components
│   ├── training/          # Training pipeline with RLHF
│   ├── evaluation/        # Metrics and analysis
│   └── utils/             # Configuration and utilities
├── scripts/
│   ├── train.py          # Full training pipeline
│   ├── evaluate.py       # Model evaluation
│   └── predict.py        # Inference on new data
├── configs/
│   ├── default.yaml      # Default configuration
│   └── ablation.yaml     # Ablation study configuration
└── tests/                # Comprehensive test suite
```

## Model Architecture

The system consists of three main components:

1. **Diffusion Caption Model**: GPT-2 based language model with diffusion noise scheduling for caption generation
2. **Reward Model**: Transformer-based model trained on preference pairs to predict caption quality
3. **Hierarchical Reward Shaper**: Novel component combining:
   - Dense rewards: Token-level CLIP alignment scores
   - Sparse rewards: Sequence-level human preference scores
   - Adaptive weighting: Confidence-based interpolation between reward types

## Custom Components

The novel hierarchical reward shaping mechanism (`HierarchicalRewardShaper`) adaptively combines:
- **CLIP-based dense rewards** for image-text alignment at token level
- **Preference-based sparse rewards** from human feedback at sequence level
- **Confidence estimation** to dynamically adjust reward weights during generation

This enables fine-grained control over caption quality dimensions including accuracy, fluency, and specificity.

## Configuration

Edit `configs/default.yaml` to customize:
- Model architecture (hidden dimensions, layers, attention heads)
- Training hyperparameters (learning rates, batch sizes, epochs)
- Reward shaping weights (CLIP weight, preference weight)
- Data settings (dataset size, tokenizer, max length)

## Results

### Training Configuration

| Setting | Value |
|---|---|
| Total Parameters | 247M (GPT-2 based) |
| Base Model Parameters | 164M |
| Reward Model Parameters | 82M |
| Hardware | NVIDIA RTX 3090 |
| Total Training Time | ~17 minutes |
| Mixed Precision | Enabled (AMP) |
| Dataset | Google Conceptual Captions |
| Training Samples | 5,000 |
| Validation Samples | 500 |

### Stage 1: Base Diffusion Caption Model Pre-training

The base model was trained for 10 epochs with a learning rate of 1e-4 and batch size 16.

| Epoch | Train Loss | Val Loss |
|---|---|---|
| 1 | 6.9664 | 5.7472 |
| 2 | 4.9661 | 4.6332 |
| 3 | 3.6093 | 2.8155 |
| 4 | 1.7532 | 1.2103 |
| 5 | 0.7012 | 0.7025 |
| 6 | 0.2665 | 0.5107 |
| 7 | 0.0885 | 0.4332 |
| 8 | 0.0289 | 0.3972 |
| 9 | 0.0127 | 0.3760 |
| 10 | 0.0091 | 0.3743 |

Best validation loss: **0.3743** (epoch 10). The train loss converging near zero while val loss plateaus around 0.37 suggests some overfitting, which is expected given the relatively small training set of 5,000 samples.

### Stage 2: Reward Model Training

The reward model was trained for 5 epochs on synthetically generated preference pairs with a learning rate of 1e-4 and batch size 16.

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 0.0053 | 0.9960 | 0.0000 | 1.0000 |
| 2 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 3-5 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |

Final accuracy: **1.0000** (train and val). The perfect accuracy reflects the fact that preference pairs were synthetically constructed with clear quality differences, making discrimination straightforward for the model. A real-world deployment would use human-annotated preferences with more nuanced distinctions.

### Stage 3: RLHF Fine-tuning with Hierarchical Rewards

RLHF fine-tuning ran for 6 out of 8 epochs before early stopping was triggered (patience=5). Learning rate was 1e-5 with batch size 8 and hierarchical reward shaping (CLIP weight=0.7, preference weight=0.3).

| Epoch | Train Loss | Train Reward | Val Loss | Val Reward |
|---|---|---|---|---|
| 1 | 0.0840 | 0.4401 | 0.3656 | **0.4835** |
| 2 | 0.0353 | 0.0816 | 0.3849 | 0.0893 |
| 3 | -0.1336 | -0.1312 | 0.6983 | 0.0731 |
| 4 | -1.5899 | -0.1657 | 2.7064 | 0.0748 |
| 5 | -4.7433 | -0.1933 | 3.5695 | 0.0734 |
| 6 | -7.0410 | -0.2109 | 4.4289 | 0.0949 |

Best validation reward: **0.4835** (epoch 1). The RLHF stage shows reward hacking behavior after epoch 1, where the policy drifts to exploit the reward model rather than genuinely improve caption quality. This is a known challenge in RLHF and would benefit from stronger KL penalties or reward model regularization in future work.

### Notes on Data and Limitations

- The training uses a subset of Google Conceptual Captions (5,000 samples), which limits the diversity of captions the model can learn.
- Preference pairs for the reward model were synthetically constructed, not human-annotated. This makes the reward model task artificially easy (perfect accuracy) and does not reflect real human preference modeling complexity.
- The RLHF fine-tuning exhibits reward hacking, a well-documented phenomenon where the policy exploits imperfections in the reward model. The best checkpoint (epoch 1) should be used for inference.
- Image features use simulated embeddings rather than a real vision encoder (e.g., CLIP), as the focus of this project is on the hierarchical reward shaping mechanism.

Evaluation results are saved to `results/training_summary.json` after training completes.

## Testing

Run the test suite:

```bash
pytest tests/
```

With coverage report:

```bash
pytest tests/ --cov=hierarchical_diffusion_rlhf_caption_refinement --cov-report=html
```

## Technical Details

The core innovation combines dense and sparse rewards with adaptive weighting:
`R_combined(t) = α(c) * R_dense(t) + (1-α(c)) * R_sparse`

Training pipeline: Pre-train base model → Train reward model → Fine-tune with RLHF.

The ablation configuration disables hierarchical shaping (CLIP weight = 0) for baseline comparison.

## Dependencies

Core dependencies:
- PyTorch 2.0+
- Transformers 4.30+
- Diffusers 0.21+
- TRL 0.7+ (for RLHF)
- Accelerate 0.20+ (for distributed training)

See `requirements.txt` for complete list.

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
