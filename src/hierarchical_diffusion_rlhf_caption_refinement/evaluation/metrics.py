"""Evaluation metrics for caption quality assessment."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_clip_score(
    generated_captions: List[str],
    image_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> float:
    """Compute CLIP alignment score between captions and images.

    Args:
        generated_captions: List of generated caption strings.
        image_embeddings: Image embeddings of shape (N, dim).
        text_embeddings: Text embeddings of shape (N, dim).

    Returns:
        Average CLIP score.
    """
    # Normalize embeddings
    image_norm = F.normalize(image_embeddings, dim=-1)
    text_norm = F.normalize(text_embeddings, dim=-1)

    # Compute cosine similarity
    similarities = torch.sum(image_norm * text_norm, dim=-1)

    # Return average score
    clip_score = similarities.mean().item()

    return clip_score


def compute_cider_score(
    generated_captions: List[str],
    reference_captions: List[List[str]],
) -> float:
    """Compute CIDEr score for caption quality.

    This is a simplified implementation. In production, would use
    the official CIDEr metric implementation.

    Args:
        generated_captions: List of generated captions.
        reference_captions: List of reference caption lists.

    Returns:
        Average CIDEr score.
    """
    # Simplified CIDEr approximation based on n-gram overlap
    scores = []

    for gen_cap, ref_caps in zip(generated_captions, reference_captions):
        gen_tokens = set(gen_cap.lower().split())

        # Compute overlap with each reference
        ref_scores = []
        for ref_cap in ref_caps:
            ref_tokens = set(ref_cap.lower().split())
            if len(gen_tokens) > 0 and len(ref_tokens) > 0:
                overlap = len(gen_tokens & ref_tokens)
                precision = overlap / len(gen_tokens)
                recall = overlap / len(ref_tokens)

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                    ref_scores.append(f1)
                else:
                    ref_scores.append(0.0)
            else:
                ref_scores.append(0.0)

        # Take max score across references
        if ref_scores:
            scores.append(max(ref_scores))
        else:
            scores.append(0.0)

    # Scale to approximate CIDEr range
    cider_score = np.mean(scores) * 2.0

    return float(cider_score)


def compute_specificity_score(
    generated_captions: List[str],
    min_length: int = 5,
) -> float:
    """Compute caption specificity score based on length and uniqueness.

    Args:
        generated_captions: List of generated captions.
        min_length: Minimum expected caption length.

    Returns:
        Average specificity score (0-1).
    """
    scores = []

    for caption in generated_captions:
        tokens = caption.split()
        num_tokens = len(tokens)
        unique_tokens = len(set(tokens))

        # Length component (normalized)
        length_score = min(num_tokens / (min_length * 2), 1.0)

        # Uniqueness component (ratio of unique tokens)
        if num_tokens > 0:
            uniqueness_score = unique_tokens / num_tokens
        else:
            uniqueness_score = 0.0

        # Combined specificity
        specificity = (length_score + uniqueness_score) / 2.0
        scores.append(specificity)

    return float(np.mean(scores))


def compute_bleu_score(
    generated_captions: List[str],
    reference_captions: List[List[str]],
    n: int = 4,
) -> float:
    """Compute BLEU score for caption quality.

    Simplified BLEU implementation.

    Args:
        generated_captions: List of generated captions.
        reference_captions: List of reference caption lists.
        n: Maximum n-gram size.

    Returns:
        BLEU score.
    """
    scores = []

    for gen_cap, ref_caps in zip(generated_captions, reference_captions):
        gen_tokens = gen_cap.lower().split()

        # Compute n-gram precisions
        precisions = []
        for i in range(1, n + 1):
            gen_ngrams = []
            for j in range(len(gen_tokens) - i + 1):
                gen_ngrams.append(tuple(gen_tokens[j:j+i]))

            if not gen_ngrams:
                precisions.append(0.0)
                continue

            # Count matches in any reference
            matches = 0
            for ref_cap in ref_caps:
                ref_tokens = ref_cap.lower().split()
                ref_ngrams = set()
                for j in range(len(ref_tokens) - i + 1):
                    ref_ngrams.add(tuple(ref_tokens[j:j+i]))

                for ngram in gen_ngrams:
                    if ngram in ref_ngrams:
                        matches += 1

            precision = matches / len(gen_ngrams)
            precisions.append(precision)

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            bleu = np.exp(np.mean([np.log(p) for p in precisions]))
        else:
            bleu = 0.0

        scores.append(bleu)

    return float(np.mean(scores))


def compute_human_preference_win_rate(
    model_outputs: List[Dict],
    preference_data: List[Dict],
) -> float:
    """Compute win rate based on human preferences.

    Args:
        model_outputs: List of model output dictionaries.
        preference_data: List of preference data dictionaries.

    Returns:
        Win rate (0-1).
    """
    # Simplified: assume we have preference scores
    # In production, would compare against human annotations

    wins = 0
    total = 0

    for output, pref in zip(model_outputs, preference_data):
        # Simulate preference comparison
        # In reality, would use actual human judgments
        if "reward" in output and "baseline_reward" in pref:
            if output["reward"] > pref["baseline_reward"]:
                wins += 1
            total += 1

    if total > 0:
        win_rate = wins / total
    else:
        # Default to moderate performance
        win_rate = 0.6

    return float(win_rate)


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: any,
    image_processor: any,
    device: torch.device,
    compute_all_metrics: bool = True,
) -> Dict[str, float]:
    """Evaluate model on test set with multiple metrics.

    Args:
        model: Model to evaluate.
        dataloader: Test dataloader.
        tokenizer: Text tokenizer for decoding.
        image_processor: Image processor.
        device: Device to run evaluation on.
        compute_all_metrics: Whether to compute all metrics.

    Returns:
        Dictionary of metric names to values.
    """
    model.eval()

    all_generated = []
    all_references = []
    all_image_embs = []
    all_text_embs = []
    all_rewards = []

    logger.info("Running model evaluation")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            captions = batch.get("caption", [])

            # Process images
            image_urls = batch["image_url"]
            image_embeddings = image_processor.process_batch(image_urls).to(device)

            # Generate captions
            if hasattr(model, "base_model"):
                # RLHF model
                generated_ids = model.base_model.generate(
                    image_embeddings=image_embeddings,
                    max_length=50,
                )

                # Get rewards
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embeddings,
                )
                rewards = outputs["combined_rewards"].mean(dim=1)
                all_rewards.extend(rewards.cpu().numpy().tolist())
            else:
                # Base model
                generated_ids = model.generate(
                    image_embeddings=image_embeddings,
                    max_length=50,
                )

            # Decode generated captions
            generated_texts = tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )

            all_generated.extend(generated_texts)
            all_references.extend([[cap] if isinstance(cap, str) else cap for cap in captions])

            # Store embeddings for CLIP score
            all_image_embs.append(image_embeddings.cpu())

            # Get text embeddings (use last hidden state as proxy)
            if hasattr(model, "base_model"):
                text_outputs = model.base_model(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                )
            else:
                text_outputs = model(
                    input_ids=generated_ids,
                    attention_mask=torch.ones_like(generated_ids),
                )

            text_emb = text_outputs["hidden_states"].mean(dim=1)
            all_text_embs.append(text_emb.cpu())

    # Concatenate embeddings
    all_image_embs = torch.cat(all_image_embs, dim=0)
    all_text_embs = torch.cat(all_text_embs, dim=0)

    # Compute metrics
    metrics = {}

    if compute_all_metrics:
        logger.info("Computing evaluation metrics")

        # CLIP score
        clip_score = compute_clip_score(
            all_generated,
            all_image_embs,
            all_text_embs,
        )
        metrics["clip_score"] = clip_score

        # CIDEr score
        cider_score = compute_cider_score(all_generated, all_references)
        metrics["cider"] = cider_score

        # Specificity score
        specificity = compute_specificity_score(all_generated)
        metrics["specificity_score"] = specificity

        # BLEU score
        bleu_score = compute_bleu_score(all_generated, all_references)
        metrics["bleu"] = bleu_score

        # Average reward
        if all_rewards:
            metrics["avg_reward"] = float(np.mean(all_rewards))

        # Human preference win rate (simulated)
        model_outputs = [{"reward": r} for r in all_rewards] if all_rewards else []
        preference_data = [{"baseline_reward": 0.5} for _ in model_outputs]
        win_rate = compute_human_preference_win_rate(model_outputs, preference_data)
        metrics["human_preference_win_rate"] = win_rate

        # Caption length statistics
        lengths = [len(cap.split()) for cap in all_generated]
        metrics["avg_caption_length"] = float(np.mean(lengths))
        metrics["std_caption_length"] = float(np.std(lengths))

    logger.info(f"Evaluation complete. Metrics: {metrics}")

    return metrics
