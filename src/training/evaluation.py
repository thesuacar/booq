"""Helpers for evaluating caption models using BLEU."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.utils.data import DataLoader


def _greedy_decode_batch(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    max_length: int,
    sos_idx: int,
    eos_idx: int,
) -> List[List[int]]:
    """Decode a batch of images into token id sequences using greedy search."""
    encoder.eval()
    decoder.eval()

    generated: List[List[int]] = [[] for _ in range(images.size(0))]

    with torch.no_grad():
        features = encoder(images.to(device))
        inputs = features.unsqueeze(1)
        states = None
        ended = torch.zeros(images.size(0), dtype=torch.bool, device=device)

        for _ in range(max_length):
            hiddens, states = decoder.lstm(inputs, states)
            logits = decoder.linear(hiddens.squeeze(1))
            predicted = logits.argmax(dim=1)

            for idx, token_id in enumerate(predicted.tolist()):
                if ended[idx]:
                    continue
                if token_id == sos_idx:
                    continue
                if token_id == eos_idx:
                    ended[idx] = True
                else:
                    generated[idx].append(token_id)

            inputs = decoder.embed(predicted).unsqueeze(1)
            if ended.all():
                break

    return generated


def _decode_reference(
    caption: torch.Tensor,
    *,
    idx_to_token: Dict[int, str],
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
) -> List[str]:
    """Convert a tensor caption into a list of tokens."""
    words: List[str] = []
    for token_id in caption.tolist():
        if token_id in (pad_idx, sos_idx):
            continue
        if token_id == eos_idx:
            break
        words.append(idx_to_token.get(token_id, "<UNK>"))
    return words


def evaluate_bleu(
    loader: DataLoader,
    *,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    device: torch.device,
    idx_to_token: Dict[int, str],
    sos_idx: int,
    eos_idx: int,
    pad_idx: int,
    max_length: int,
    max_batches: Optional[int] = None,
) -> float:
    """Compute corpus-level BLEU score over a dataloader."""
    smoothing = SmoothingFunction().method1
    scores: List[float] = []

    for batch_idx, (images, captions) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        predictions = _greedy_decode_batch(
            encoder,
            decoder,
            images,
            device=device,
            max_length=max_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
        )

        for generated_ids, reference_tensor in zip(predictions, captions):
            reference = _decode_reference(
                reference_tensor,
                idx_to_token=idx_to_token,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                pad_idx=pad_idx,
            )
            candidate = [idx_to_token.get(tid, "<UNK>") for tid in generated_ids]

            if not reference or not candidate:
                continue

            score = sentence_bleu(
                [reference],
                candidate,
                smoothing_function=smoothing,
                weights=(0.25, 0.25, 0.25, 0.25),
            )
            scores.append(score)

    return float(sum(scores) / len(scores)) if scores else 0.0
