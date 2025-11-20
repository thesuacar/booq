"""
Training utilities for Image → Caption model.
Handles:
- Data loading
- Training
- BLEU evaluation
- Early stopping
- Saving best & final model (joblib + pytorch)

This file is called from: `python -m src.training.model_training`
"""

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from src.training.preprocessing_utils import preprocess_dataset, clean_captions_txt
from src.training.encoders import EncoderCNN, DecoderRNN
from src.training.evaluation import evaluate_bleu


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------
# Config structure (used from model_training.py)
# --------------------------------------------------------
@dataclass
class TrainConfig:
    caption_txt: str
    image_dir: str
    embed_size: int = 256
    hidden_size: int = 512
    vocab_size: int = 5000
    batch_size: int = 32
    lr: float = 0.001
    num_epochs: int = 20
    patience: int = 5            # early stopping patience
    eval_every: int = 1
    save_dir: str = "runs"       # outputs: best.pth & model.joblib


# --------------------------------------------------------
# TRAINING FUNCTION
# --------------------------------------------------------
def train_with_config(cfg: TrainConfig):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n Loading dataset...")
    df = clean_captions_txt(cfg.image_dir, cfg.caption_txt)
    train_ds, dev_ds, _, vocab = preprocess_dataset(df, cfg.image_dir, vocab_size=cfg.vocab_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size)

    inv_vocab = {v: k for k, v in vocab.items()}

    encoder = EncoderCNN(cfg.embed_size).to(DEVICE)
    decoder = DecoderRNN(cfg.embed_size, cfg.hidden_size, len(vocab)).to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)

    best_bleu = -1
    patience_counter = 0

    print("\n Training started...")

    for epoch in range(1, cfg.num_epochs + 1):
        encoder.train()
        decoder.train()

        running_loss = 0

        for imgs, caps in train_loader:
            imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)

            optimizer.zero_grad()

            feats = encoder(imgs)
            outputs = decoder(feats, caps[:, :-1])  # predict next token

            outputs = outputs[:, 1:, :]  # (batch, seq_len, vocab)

            loss = loss_fn(
                outputs.reshape(-1, outputs.size(-1)),
                caps[:, 1:].reshape(-1)
            )

            loss.backward()
            clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        if epoch % cfg.eval_every == 0:
            bleu = evaluate_bleu(
                loader=dev_loader,
                encoder=encoder,
                decoder=decoder,
                device=DEVICE,
                idx_to_token=inv_vocab,
                sos_idx=1, eos_idx=2, pad_idx=0,
                max_length=20,
                max_batches=6,
            )

            print(f"Epoch {epoch}/{cfg.num_epochs} | Loss: {avg_loss:.4f} | BLEU: {bleu:.4f}")

            # Track best BLEU
            if bleu > best_bleu:
                best_bleu = bleu
                patience_counter = 0

                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "vocab": vocab,
                        "epoch": epoch,
                    },
                    save_dir / "best.pth",
                )

            else:
                patience_counter += 1

            if patience_counter >= cfg.patience:
                print(f"\n Early stopping triggered. Best BLEU = {best_bleu:.4f}")
                break

    # --------------------------------------------------------
    # Save final full model + vocab as joblib
    # --------------------------------------------------------
    print("\n Saving final model...")

    joblib.dump(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "vocab": vocab,
        },
        save_dir / "model.joblib",
    )

    print("Training completed.\n")
