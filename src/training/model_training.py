from src.preprocessing.preprocessing_utils import clean_captions_txt, preprocess_dataset
from src.training.training_utils import train, evaluate_bleu
from src.training.encoders import EncoderCNN, DecoderRNN
import torch


def main():
    project_root = Path(__file__).resolve().parents[2]
    image_dir = project_root / "data" / "archive30k" / "Images"
    caption_file = project_root / "data" / "archive30k" / "captions.txt"

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
