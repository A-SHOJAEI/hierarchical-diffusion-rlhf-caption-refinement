"""Hierarchical Diffusion RLHF Caption Refinement.

A two-stage generative system for image caption generation and refinement
using diffusion models and RLHF with hierarchical reward shaping.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"
__license__ = "MIT"

from hierarchical_diffusion_rlhf_caption_refinement.models.model import (
    DiffusionCaptionModel,
    RLHFCaptionRefiner,
)
from hierarchical_diffusion_rlhf_caption_refinement.training.trainer import (
    HierarchicalRLHFTrainer,
)

__all__ = [
    "DiffusionCaptionModel",
    "RLHFCaptionRefiner",
    "HierarchicalRLHFTrainer",
]
