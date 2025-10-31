from pathlib import Path
from src.training.training_utils import train
import kagglehub
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# Download latest version of Flickr8k
dataset_path = kagglehub.dataset_download("adityajn105/flickr8k")
dataset_path = Path(dataset_path)

print("Dataset downloaded to:", dataset_path)

def main():
    # Correct dataset paths
    image_dir = dataset_path / "Images"
    caption_file = dataset_path / "captions.txt"

    # Validate paths
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not caption_file.exists():
        raise FileNotFoundError(f"Captions file not found: {caption_file}")

    train(
        caption_txt=str(caption_file),
        image_dir=str(image_dir),
        vocab_size=5000,
        embed_size=256,
        hidden_size=512,
        lr=0.001,
        num_epochs=10,
        eval_every=1,
    )

if __name__ == "__main__":
    main()
