import os
import csv
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer, word_tokenize

class FlickrDataset(Dataset):
    def __init__(self, df, img_path, transform=None):
        self.df = df
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_path, row['image'])
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        caption = torch.tensor(row['encoded'], dtype=torch.long)
        raw_caption = row.get('caption', "")
        return img, caption, raw_caption


def clean_captions_txt(image_path, caption_path):

    image_path = os.fspath(image_path)
    caption_path = os.fspath(caption_path)

    if not os.path.isfile(caption_path):
        raise FileNotFoundError(f"Caption file not found at {caption_path}")

    if not os.path.isdir(image_path):
        raise FileNotFoundError(f"Image directory not found at {image_path}")

    data = []

    with open(caption_path, "r", encoding="utf-8", newline="") as f:
        first_non_empty = None
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped:
                first_non_empty = stripped
                break

        f.seek(0)

        delimiter = "\t" if first_non_empty and "\t" in first_non_empty else ","
        reader = csv.reader(f, delimiter=delimiter)

        for row in reader:
            if not row:
                continue

            first_field = row[0].strip()
            if not first_field:
                continue

            # Skip header rows if present
            if first_field.lower() in {"image", "image_id", "filename"}:
                continue

            if len(row) < 2:
                continue

            caption_parts = row[1:]
            caption = caption_parts[0].strip() if caption_parts else ""
            if len(caption_parts) > 1:
                joiner = " " if delimiter == "\t" else ", "
                caption = joiner.join(part.strip() for part in caption_parts).strip()

            if not caption:
                continue

            img = first_field.split("#")[0].strip().lower()
            data.append((img, caption))

    df = pd.DataFrame(data, columns=["image", "caption"])

    if df.empty:
        print("No valid caption-image pairs found after parsing.")
        return df

    available = {f.lower(): f for f in os.listdir(image_path)}
    df["image"] = df["image"].map(lambda x: available.get(x))

    df = df.dropna(subset=["image"]).reset_index(drop=True)

    print(f"Total valid image entries: {len(df)}")
    print(f"Unique usable images: {df['image'].nunique()}")

    return df



def build_vocab(token_lists, vocab_size=1000):
    tokens = [t.lower() for lst in token_lists for t in lst]
    freq = Counter(tokens).most_common(vocab_size)

    vocab = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
    }
    for i, (word, _) in enumerate(freq, start=4):
        vocab[word] = i

    return vocab


def encode(tokens, vocab, max_len=15):
    ids = [vocab.get("<SOS>")]
    for t in tokens[: max_len - 2]:
        ids.append(vocab.get(t.lower(), vocab["<UNK>"]))
    ids.append(vocab.get("<EOS>"))

    # Padding
    ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


def preprocess_dataset(df, images_path, vocab_size=1000, sample_size=None):
    """Tokenize, build vocab, encode, split dataset & create DataLoaders"""

    if sample_size:
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    def safe_tokenize(text):
        """Tokenize text, preferring punkt if it is already installed."""
        if not hasattr(safe_tokenize, "_tokenizer"):
            tokenizer = TreebankWordTokenizer().tokenize
            try:
                word_tokenize("NLTK probe")
            except LookupError:
                print("NLTK 'punkt' tokenizer not available. Falling back to TreebankWordTokenizer.")
            else:
                tokenizer = word_tokenize
            safe_tokenize._tokenizer = tokenizer

        return safe_tokenize._tokenizer(text)

    # Tokenize captions
    df['tokens'] = df['caption'].astype(str).apply(safe_tokenize)

    vocab = build_vocab(df['tokens'], vocab_size=vocab_size)

    df['encoded'] = df['tokens'].apply(lambda x: encode(x, vocab))

    # Train/Dev/Test split
    train = df.sample(frac=0.7, random_state=42)
    temp = df.drop(train.index)
    dev = temp.sample(frac=0.5, random_state=42)
    test = temp.drop(dev.index)

    # Reset indexes for DataLoaders
    train, dev, test = train.reset_index(drop=True), dev.reset_index(drop=True), test.reset_index(drop=True)

    train_ds = FlickrDataset(train, images_path, transform)
    dev_ds = FlickrDataset(dev, images_path, transform)
    test_ds = FlickrDataset(test, images_path, transform)

    return train_ds, dev_ds, test_ds, vocab
