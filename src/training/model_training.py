"""
Training entry point for Image → Caption model.
THIS IS THE ONLY SCRIPT YOU RUN TO TRAIN THE MODEL.

Usage:
    (.venv) ➜ python -m src.training.model_training
"""

from pathlib import Path
from src.training.training_utils import TrainConfig, train_with_config
import kagglehub


def main():
    # Download dataset (Flickr8k) using KaggleHub
    dataset_path = kagglehub.dataset_download("adityajn105/flickr8k")
    dataset_path = Path(dataset_path)   # <--- convert to Path()

    config = TrainConfig(
        caption_txt=str(dataset_path / "captions.txt"),
        image_dir=str(dataset_path / "Images"),  # dataset folder uses `Images`
        embed_size=256,
        hidden_size=512,
        vocab_size=5000,
        batch_size=32,
        lr=0.001,
        num_epochs=20,
        patience=5,             # early stopping patience
        eval_every=1,
        save_dir=str(Path.cwd() / "runs")  # save under project root
    )

    print("\n Starting training with config:")
    for k, v in config.__dict__.items():
        print(f"  {k}: {v}")

    train_with_config(config)

    print("Training complete!, model saved to 'runs/' directory.")


if __name__ == "__main__":
    main()
