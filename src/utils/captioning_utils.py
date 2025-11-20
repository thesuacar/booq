# src/utils/captioning_utils.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------------------------------------------
# Model definitions
# -------------------------------------------------------------------

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
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        outputs, _ = self.lstm(embeddings)
        return self.fc(outputs)


# -------------------------------------------------------------------
# Captioner wrapper
# -------------------------------------------------------------------

@dataclass
class _CaptionConfig:
    embed_size: int = 256
    hidden_size: int = 512
    sos_idx: int = 1
    eos_idx: int = 2
    pad_idx: int = 0
    max_length: int = 20


class ImageCaptioner:
    """
    Loads the trained CNN+LSTM model and generates captions.

    Exposed API:
        generate_batch(List[Path]) -> List[str]
    """

    def __init__(
        self,
        model_dir: Optional[Path | str] = None,
        checkpoint_path: Optional[Path | str] = None,
    ):
        # Determine checkpoint path
        if checkpoint_path is not None:
            self.ckpt_path = Path(checkpoint_path)
        else:
            if model_dir is None:
                # project_root/src/utils -> project_root/src
                project_root = Path(__file__).resolve().parents[1]
                model_dir = project_root / "training"
            self.ckpt_path = Path(model_dir) / "best_epoch.pth"

        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"Captioning checkpoint not found: {self.ckpt_path}")

        # Base config
        self.cfg = _CaptionConfig()

        # Load vocab + encoder + decoder
        self._load_checkpoint()

        # Build transforms LAST
        self._build_transforms()

    # ---------------- Internals ----------------

    def _build_transforms(self):
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225),
                ),
            ]
        )

    def _load_checkpoint(self):
        ckpt = torch.load(self.ckpt_path, map_location=DEVICE)

        self.vocab = ckpt["vocab"]
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

        # Update config from checkpoint if present
        cfg_dict = ckpt.get("config")
        if cfg_dict:
            self.cfg.embed_size = cfg_dict.get("embed_size", self.cfg.embed_size)
            self.cfg.hidden_size = cfg_dict.get("hidden_size", self.cfg.hidden_size)

        # Build encoder + decoder
        self.encoder = EncoderCNN(self.cfg.embed_size).to(DEVICE)
        self.decoder = DecoderRNN(
            self.cfg.embed_size,
            self.cfg.hidden_size,
            vocab_size=len(self.vocab),
        ).to(DEVICE)

        self.encoder.load_state_dict(ckpt["encoder"])
        self.decoder.load_state_dict(ckpt["decoder"])

        self.encoder.eval()
        self.decoder.eval()

    # ---------------- Decoding ----------------

    def _decode_single(self, image: Image.Image) -> str:
        img_tensor = self.transform(image).unsqueeze(0).to(DEVICE)

        sos = self.cfg.sos_idx
        eos = self.cfg.eos_idx
        pad = self.cfg.pad_idx

        with torch.no_grad(), torch.amp.autocast(
            device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")
        ):
            features = self.encoder(img_tensor)
            inputs = features.unsqueeze(1)
            states = None
            generated_ids: List[int] = []

            for _ in range(self.cfg.max_length):
                out, states = self.decoder.lstm(inputs, states)
                logits = self.decoder.fc(out.squeeze(1))
                predicted = logits.argmax(dim=1)
                token_id = int(predicted.item())

                if token_id == eos:
                    break
                if token_id not in (pad, sos):
                    generated_ids.append(token_id)

                inputs = self.decoder.embed(predicted).unsqueeze(1)

        tokens = [self.inv_vocab.get(t, "<UNK>") for t in generated_ids]
        caption = " ".join(tokens).strip()
        return caption or "(no caption)"

    # ---------------- Public API ----------------

    def generate_batch(self, image_paths: List[Path]) -> List[str]:
        captions: List[str] = []
        for p in image_paths:
            try:
                img = Image.open(p).convert("RGB")
                captions.append(self._decode_single(img))
            except Exception as exc:
                print(f"[ImageCaptioner] Failed on {p}: {exc}")
                captions.append("(caption unavailable)")
        return captions

