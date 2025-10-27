from src.preprocessing.preprocessing_utils import clean_captions_txt, preprocess_dataset
from src.training.training_utils import train, evaluate_bleu
from src.training.encoders import EncoderCNN, DecoderRNN
import torch
import kagglehub

# Download latest version
path = kagglehub.dataset_download("adityajn105/flickr8k")

print("Path to dataset files:", path)

def main():

    # Paths
    image_path = "data/Images/"
    caption_path = "data/captions.txt"

    #Preprocess using the official pipeline
    df = clean_captions_txt(image_path, caption_path)
    train_ds, dev_ds, test_ds, vocab = preprocess_dataset(df, image_path, vocab_size=5000)

    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=32)

    #Initialize models
    encoder = EncoderCNN(embed_size=256).to(device)
    decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=vocab_size).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab["<PAD>"])
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    # Train
    train(
        encoder,
        decoder,
        train_loader,
        dev_loader,
        criterion,
        optimizer,
        num_epochs=10,
        vocab_size=vocab_size,
        evaluate_bleu_fn=lambda: evaluate_bleu(dev_loader, encoder, decoder, vocab, device),
        eval_interval=1,
        device=device,
    )


if __name__ == "__main__":
    main()
