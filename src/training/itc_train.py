#!/usr/bin/env python
# coding: utf-8

"""
Image → Caption Training (Flickr8k) with:
- CNN (ResNet18) + LSTM captioning model
- tqdm progress bars (per batch)
- CarbonTracker energy & CO₂ logging
- Early stopping and BLEU evaluation
- Best-epoch selection + export

Designed to run on GPU4EDU via SLURM and integrate with the booq project.
"""

import os
import csv
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

import pandas as pd
from PIL import Image

import kagglehub
from tqdm.auto import tqdm

import nltk
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torchvision.transforms as transforms

# ---- CarbonTracker (optional, but recommended) ----
try:
    from carbontracker.tracker import CarbonTracker
    CARBONTRACKING_AVAILABLE = True
except ImportError:
    CarbonTracker = None
    CARBONTRACKING_AVAILABLE = False

# ------------------------------------------------------------------
# Global config
# ------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# Central place where models are stored in the repo
IMAGE_CAPTIONING_MODEL = Path("src/training/best_epoch")

# Make sure tokenizer data is available
nltk.download("punkt", quiet=True)
try:
    nltk.download("punkt_tab", quiet=True)
except Exception:
    # Not critical if this fails
    pass


# ----------------------------------------------------
# Dataset & preprocessing
# ----------------------------------------------------

class FlickrDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_path: str, transform=None):
        self.df = df
        self.img_path = img_path
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_path, row["image"])).convert("RGB")

        if self.transform:
            img = self.transform(img)

        caption = torch.tensor(row["encoded"], dtype=torch.long)
        return img, caption


def clean_captions_txt(image_path: str, caption_path: str) -> pd.DataFrame:
    """Load and normalise the captions file (Flickr8k-style)."""
    if not os.path.isfile(caption_path):
        raise FileNotFoundError(caption_path)

    rows = []
    with open(caption_path, "r", encoding="utf-8") as f:
        first = f.readline()
        delim = "\t" if "\t" in first else ","
        f.seek(0)
        reader = csv.reader(f, delimiter=delim)

        for row in reader:
            if not row or row[0].lower() in {"image", "filename"}:
                continue

            img = row[0].split("#")[0].strip().lower()
            caption = row[1].strip()
            rows.append((img, caption))

    df = pd.DataFrame(rows, columns=["image", "caption"])

    lookup = {f.lower(): f for f in os.listdir(image_path)}
    df["image"] = df["image"].apply(lambda x: lookup.get(x))
    df = df.dropna().reset_index(drop=True)
    return df


def _tokenizer():
    try:
        nltk.data.find("tokenizers/punkt/english.pickle")
    except LookupError:
        nltk.download("punkt", quiet=True)
    return word_tokenize


def build_vocab(token_lists, vocab_size: int = 5000) -> Dict[str, int]:
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    all_tokens = [t.lower() for lst in token_lists for t in lst]
    freq = Counter(all_tokens).most_common(vocab_size - 4)
    vocab.update({w: idx + 4 for idx, (w, _) in enumerate(freq)})
    return vocab


def encode(tokens: List[str], vocab: Dict[str, int], max_len: int = 20) -> List[int]:
    ids = [vocab["<SOS>"]] + [vocab.get(t.lower(), vocab["<UNK>"]) for t in tokens[: max_len - 2]]
    ids.append(vocab["<EOS>"])
    ids.extend([vocab["<PAD>"]] * (max_len - len(ids)))
    return ids


def preprocess_dataset(
    df: pd.DataFrame,
    images_path: str,
    vocab_size: int = 5000,
    sample_size: Optional[int] = None,
):
    tok = _tokenizer()

    if sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    df["tokens"] = df["caption"].astype(str).apply(tok)
    vocab = build_vocab(df["tokens"], vocab_size)
    df["encoded"] = df["tokens"].apply(lambda x: encode(x, vocab, max_len=20))

    train = df.sample(frac=0.7, random_state=42)
    temp = df.drop(train.index)
    dev = temp.sample(frac=0.5, random_state=42)
    test = temp.drop(dev.index)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.485, 0.456, 0.406),
                (0.229, 0.224, 0.225),
            ),
        ]
    )

    return (
        FlickrDataset(train.reset_index(drop=True), images_path, transform),
        FlickrDataset(dev.reset_index(drop=True), images_path, transform),
        FlickrDataset(test.reset_index(drop=True), images_path, transform),
        vocab,
    )


# ----------------------------------------------------
# Encoder & Decoder
# ----------------------------------------------------

class EncoderCNN(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.norm = nn.LayerNorm(embed_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(images).squeeze()

        features = self.fc(features)
        return self.norm(features)


class DecoderRNN(nn.Module):
    def __init__(
        self,
        embed_size: int,
        hidden_size: int,
        vocab_size: int,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(embeddings)
        return self.fc(outputs)


# ----------------------------------------------------
# BLEU Evaluation (greedy decoding)
# ----------------------------------------------------

def _greedy_decode_batch(
    encoder,
    decoder,
    images,
    *,
    device,
    max_length,
    sos_idx,
    eos_idx,
):
    encoder.eval()
    decoder.eval()

    generated = [[] for _ in range(images.size(0))]
    ended = torch.zeros(images.size(0), dtype=torch.bool, device=device)

    with torch.no_grad(), torch.amp.autocast(
        device_type=getattr(device, "type", "cuda"),
        enabled=(getattr(device, "type", "cuda") == "cuda"),
    ):
        features = encoder(images.to(device))
        inputs = features.unsqueeze(1)
        states = None

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
            if ended.all():
                break

    return generated


def evaluate_bleu(
    loader,
    *,
    encoder,
    decoder,
    device,
    idx_to_token,
    sos_idx,
    eos_idx,
    pad_idx,
    max_length,
    max_batches=None,
):
    smoothing = SmoothingFunction().method1
    scores = []

    for batch_idx, (images, captions) in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        preds = _greedy_decode_batch(
            encoder,
            decoder,
            images,
            device=device,
            max_length=max_length,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
        )

        for pred_ids, caption_ids in zip(preds, captions):
            target = [
                idx_to_token.get(int(t), "<UNK>")
                for t in caption_ids.tolist()
                if t not in (pad_idx, sos_idx)
            ]
            generated = [idx_to_token.get(t, "<UNK>") for t in pred_ids]
            if generated:
                scores.append(
                    sentence_bleu(
                        [target],
                        generated,
                        smoothing_function=smoothing,
                    )
                )

    return sum(scores) / len(scores) if scores else 0.0


# ----------------------------------------------------
# Training utilities with tqdm + CarbonTracker
# ----------------------------------------------------

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
    patience: int = 20
    eval_every: int = 1
    # default to repo-local model dir
    save_dir: str = str(IMAGE_CAPTIONING_MODEL)


def train_with_config(cfg: TrainConfig):
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1] Loading dataset...")
    df = clean_captions_txt(cfg.image_dir, cfg.caption_txt)
    train_ds, dev_ds, _, vocab = preprocess_dataset(
        df,
        cfg.image_dir,
        vocab_size=cfg.vocab_size,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size)

    inv_vocab = {v: k for k, v in vocab.items()}

    print("\n[Step 2] Building models on", DEVICE)
    encoder = EncoderCNN(cfg.embed_size).to(DEVICE)
    decoder = DecoderRNN(cfg.embed_size, cfg.hidden_size, len(vocab)).to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = AdamW(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg.lr,
    )

    best_bleu = -1.0
    patience_counter = 0

    # ----- CarbonTracker setup -----
    tracker = None
    if CARBONTRACKING_AVAILABLE:
        carbon_log_dir = save_dir / "carbon_logs"
        carbon_log_dir.mkdir(parents=True, exist_ok=True)
        tracker = CarbonTracker(
            epochs=cfg.num_epochs,
            components="gpu",
            log_dir=str(carbon_log_dir),
        )
        print(f"\n[CarbonTracker] Logging energy usage to: {carbon_log_dir}")
    else:
        print(
            "\n[CarbonTracker] Not installed. Run `pip install carbontracker` "
            "inside your 'booq' environment to enable energy logging."
        )

    print("\n[Step 3] Training loop starting...\n")

    try:
        # Track BLEU history for top-5 ranking
        bleu_history = []

        for epoch in range(1, cfg.num_epochs + 1):
            epoch_start = time.time()

            if tracker is not None:
                tracker.epoch_start()

            encoder.train()
            decoder.train()
            running_loss = 0.0

            batch_pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{cfg.num_epochs}",
                leave=False,
            )

            for imgs, caps in batch_pbar:
                imgs, caps = imgs.to(DEVICE), caps.to(DEVICE)

                optimizer.zero_grad()

                feats = encoder(imgs)
                outputs = decoder(feats, caps[:, :-1])
                outputs = outputs[:, 1:, :]

                loss = loss_fn(
                    outputs.reshape(-1, outputs.size(-1)),
                    caps[:, 1:].reshape(-1),
                )

                loss.backward()
                clip_grad_norm_(
                    list(encoder.parameters()) + list(decoder.parameters()),
                    1.0,
                )
                optimizer.step()

                running_loss += loss.item()
                batch_pbar.set_postfix(loss=loss.item())

            avg_loss = running_loss / len(train_loader)

            # ---- Evaluation ----
            bleu = float("nan")
            if epoch % cfg.eval_every == 0:
                encoder.eval()
                decoder.eval()
                bleu = evaluate_bleu(
                    loader=dev_loader,
                    encoder=encoder,
                    decoder=decoder,
                    device=DEVICE,
                    idx_to_token=inv_vocab,
                    sos_idx=1,
                    eos_idx=2,
                    pad_idx=0,
                    max_length=20,
                    max_batches=6,
                )

            # Save BLEU history
            bleu_history.append((epoch, float(avg_loss), float(bleu)))

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch}/{cfg.num_epochs} "
                f"| Time: {epoch_time:.1f}s "
                f"| Loss: {avg_loss:.4f} "
                f"| BLEU: {bleu:.4f}"
            )

            # ---- 1) Save every epoch (with config) ----
            epoch_ckpt = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab": vocab,
                "epoch": epoch,
                "bleu": float(bleu),
                "loss": float(avg_loss),
                "config": asdict(cfg),
            }

            torch.save(epoch_ckpt, save_dir / f"epoch_{epoch}.pth")
            print(f"  ↳ Saved checkpoint epoch_{epoch}.pth")

            # ---- Best BLEU during training ----
            if not torch.isnan(torch.tensor(bleu)) and bleu > best_bleu:
                best_bleu = bleu
                patience_counter = 0

                torch.save(epoch_ckpt, save_dir / "best.pth")
                print(f"  ↳ New best BLEU: {best_bleu:.4f} (model saved)")
            else:
                patience_counter += 1
                print(
                    f"  ↳ No improvement. Patience: "
                    f"{patience_counter}/{cfg.patience}"
                )

            if tracker is not None:
                tracker.epoch_end()

            if patience_counter >= cfg.patience:
                print(f"\n[Early Stopping] Best BLEU = {best_bleu:.4f}")
                break

    finally:
        # Always stop tracker
        if tracker is not None:
            tracker.stop()

        print("\n[Step 4] Ranking top 5 epochs by BLEU...")

        # ---- 2) Create a sorted top-5 BLEU table ----
        sorted_epochs = sorted(bleu_history, key=lambda x: x[2], reverse=True)
        top5 = sorted_epochs[:5]

        # Save as a nice text file
        table_path = save_dir / "top5_epochs.txt"
        with open(table_path, "w") as f:
            f.write("Epoch | Loss | BLEU\n")
            f.write("---------------------\n")
            for (ep, ls, bl) in top5:
                f.write(f"{ep:5d} | {ls:.4f} | {bl:.4f}\n")

        print(f"Top 5 BLEU epochs saved in: {table_path}")

        # ---- 3) Auto-load best epoch & save cleanly as best_epoch.pth ----
        best_epoch = top5[0][0]
        best_epoch_file = save_dir / f"epoch_{best_epoch}.pth"
        best_epoch_export = save_dir / "best_epoch.pth"

        print(f"[Step 5] Best epoch determined: {best_epoch}")
        print(f"Copying epoch_{best_epoch}.pth → best_epoch.pth")

        best_ckpt = torch.load(best_epoch_file, map_location=DEVICE)
        # Ensure config is present
        if "config" not in best_ckpt:
            best_ckpt["config"] = asdict(cfg)

        torch.save(best_ckpt, best_epoch_export)

        print("\n[Step 6] Saving final model...")
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "vocab": vocab,
                "config": asdict(cfg),
            },
            save_dir / "final.pth",
        )
        print(f"Training completed. Models saved to: {save_dir}")


# ----------------------------------------------------
# Main: download dataset & launch training (demo mode)
# ----------------------------------------------------

def run_flickr8k_demo():
    """Demo entrypoint that downloads Flickr8k via kagglehub and trains."""
    print("\n[Setup] Downloading Flickr8k dataset (if not already cached)...")
    dataset_path = kagglehub.dataset_download("adityajn105/flickr8k")
    dataset_path = Path(dataset_path)

    config = TrainConfig(
        caption_txt=str(dataset_path / "captions.txt"),
        image_dir=str(dataset_path / "Images"),
        # save inside repo path by default
        save_dir=str(IMAGE_CAPTIONING_MODEL),
    )

    print("\n[Config] Starting training with configuration:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    train_with_config(config)


if __name__ == "__main__":
    run_flickr8k_demo()
