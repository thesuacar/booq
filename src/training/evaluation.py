# src/training/bleu.py
from typing import Dict, List, Optional
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def _greedy_decode_batch(encoder, decoder, images, *, device, max_length, sos_idx, eos_idx):
    encoder.eval()
    decoder.eval()

    with torch.no_grad(), torch.cuda.amp.autocast(device.type == "cuda"):
        features = encoder(images.to(device))
        inputs = features.unsqueeze(1)
        states = None
        generated = [[] for _ in range(images.size(0))]
        ended = torch.zeros(images.size(0), dtype=torch.bool, device=device)

        for _ in range(max_length):
            out, states = decoder.lstm(inputs, states)
            logits = decoder.fc(out.squeeze(1))
            predicted = logits.argmax(dim=1)

            for i, token in enumerate(predicted.tolist()):
                if not ended[i] and token != sos_idx:
                    if token == eos_idx:
                        ended[i] = True
                    else:
                        generated[i].append(token)

            inputs = decoder.embed(predicted).unsqueeze(1)
            if ended.all(): break

    return generated


def evaluate_bleu(loader, *, encoder, decoder, device, idx_to_token, sos_idx, eos_idx, pad_idx, max_length, max_batches=None):
    smoothing = SmoothingFunction().method1
    scores = []

    for batch_idx, (images, captions) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        preds = _greedy_decode_batch(
            encoder, decoder, images,
            device=device,
            max_length=max_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
        )

        for pred_ids, caption_ids in zip(preds, captions):
            target = [idx_to_token.get(int(t), "<UNK>")
                      for t in caption_ids.tolist()
                      if t not in (pad_idx, sos_idx)]
            generated = [idx_to_token.get(t, "<UNK>") for t in pred_ids]
            if generated:
                scores.append(sentence_bleu([target], generated, smoothing_function=smoothing))

    return sum(scores) / len(scores) if scores else 0.0
