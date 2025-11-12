# src/preprocessing/preprocessing_utils.py
import os
import csv
import pandas as pd
from PIL import Image
from collections import Counter
from typing import List, Dict, Tuple, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import nltk
from nltk.tokenize import word_tokenize, TreebankWordTokenizer


class FlickrDataset(Dataset):
    def __init__(self, df, img_path, transform=None):
        self.df = df
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_path, row["image"])).convert("RGB")

        if self.transform:
            img = self.transform(img)

        caption = torch.tensor(row["encoded"], dtype=torch.long)
        return img, caption


def clean_captions_txt(image_path, caption_path):
    if not os.path.isfile(caption_path):
        raise FileNotFoundError(caption_path)

    rows = []
    with open(caption_path, "r", encoding="utf-8") as f:
        delim = "\t" if "\t" in f.readline() else ","
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
        return word_tokenize
    except LookupError:
        nltk.download("punkt", quiet=True)
        return word_tokenize


def build_vocab(token_lists, vocab_size=5000):
    vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
    freq = Counter([t.lower() for lst in token_lists for t in lst]).most_common(vocab_size - 4)
    vocab.update({w: idx + 4 for idx, (w, _) in enumerate(freq)})
    return vocab


def encode(tokens: List[str], vocab: Dict[str, int], max_len=20):
    ids = [vocab["<SOS>"]] + [vocab.get(t.lower(), vocab["<UNK>"]) for t in tokens[: max_len - 2]]
    ids.append(vocab["<EOS>"])
    ids.extend([vocab["<PAD>"]] * (max_len - len(ids)))
    return ids


def preprocess_dataset(df, images_path, vocab_size=5000, sample_size=None):
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

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])

    return (
        FlickrDataset(train.reset_index(drop=True), images_path, transform),
        FlickrDataset(dev.reset_index(drop=True), images_path, transform),
        FlickrDataset(test.reset_index(drop=True), images_path, transform),
        vocab,
    )
