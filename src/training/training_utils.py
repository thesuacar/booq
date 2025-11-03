# src/training/training_utils.py
from src.preprocessing.preprocessing_utils import preprocess_dataset, clean_captions_txt
from src.training.encoders import EncoderCNN, DecoderRNN
from src.training.evaluation import evaluate_bleu

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from dataclasses import dataclass
import joblib
import os
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class TrainConfig:
    caption_txt: str
    image_dir: str
    embed_size: int = 256
    hidden_size: int = 512
    vocab_size: int = 5000
    batch_size: int = 64
    lr: float = 1e-3
    num_epochs: int = 20
    patience: int = 5
    eval_every: int = 1
    save_dir: str = "runs"


def train_with_config(cfg: TrainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)

    df = clean_captions_txt(cfg.image_dir, cfg.caption_txt)
    train_ds, dev_ds, test_ds, vocab = preprocess_dataset(df, cfg.image_dir, cfg.vocab_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size)

    encoder = EncoderCNN(cfg.embed_size).to(device)
    decoder = DecoderRNN(cfg.embed_size, cfg.hidden_size, len(vocab)).to(device)

    optimizer = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=cfg.lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

    best_bleu, patience = -1, 0

    for epoch in range(1, cfg.num_epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = 0

        for imgs, caps in train_loader:
            imgs, caps = imgs.to(device), caps.to(device)
            optimizer.zero_grad()

            feats = encoder(imgs)
            outputs = decoder(feats, caps[:, :-1])
            loss = loss_fn(outputs.reshape(-1, len(vocab)), caps[:, 1:].reshape(-1))
            loss.backward()
            clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            optimizer.step()

            total_loss += loss.item()

        if epoch % cfg.eval_every == 0:
            bleu = evaluate_bleu(
                loader=dev_loader,
                encoder=encoder,
                decoder=decoder,
                device=device,
                idx_to_token={v: k for k, v in vocab.items()},
                sos_idx=1, eos_idx=2, pad_idx=0,
                max_length=20, max_batches=5,
            )

            print(f"Epoch {epoch}/{cfg.num_epochs} | Loss: {total_loss:.4f} | BLEU: {bleu:.4f}")

            if bleu > best_bleu:
                best_bleu = bleu
                patience = 0
                torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                           f"{cfg.save_dir}/best.pth")
            else:
                patience += 1

            if patience >= cfg.patience:
                print("Early stopping triggered.")
                break

    joblib.dump(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "vocab": vocab},
        f"{cfg.save_dir}/model.joblib"
    )

    print(f"✅ Training complete. Best BLEU={best_bleu:.4f}")
