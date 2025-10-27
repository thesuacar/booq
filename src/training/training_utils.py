import time
import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from src.preprocessing.preprocessing_utils import clean_captions_txt, preprocess_dataset
from src.training.encoders import EncoderCNN, DecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smooth = SmoothingFunction().method1


def evaluate_bleu(loader, encoder, decoder, inv_vocab, device, n=5):
    encoder.eval()
    decoder.eval()
    scores = []

    with torch.no_grad():
        for i, (imgs, caps, txt) in enumerate(loader):
            if i >= n: break
            imgs, caps = imgs.to(device), caps.to(device)

            feats = encoder(imgs)
            outputs = decoder(feats, caps[:, :-1])
            preds = outputs.argmax(2).cpu().numpy()
            refs = caps.cpu().numpy()

            for r, p in zip(refs, preds):
                ref_caption = [inv_vocab.get(idx, '') for idx in r if idx > 3]
                hyp_caption = [inv_vocab.get(idx, '') for idx in p if idx > 3]

                if hyp_caption:
                    score = sentence_bleu(
                        [ref_caption], hyp_caption,
                        smoothing_function=smooth,
                        weights=(0.25, 0.25, 0.25, 0.25)
                    )
                    scores.append(score)

    return sum(scores) / len(scores) if scores else 0


def train(
    caption_txt="captions.txt",
    image_dir="Images/",
    vocab_size=1000,
    embed_size=256,
    hidden_size=512,
    lr=0.001,
    num_epochs=50,
    eval_every=10,
):

    print("Loading & preprocessing data...")
    df = clean_captions_txt(image_dir, caption_txt)
    train_ds, dev_ds, test_ds, vocab = preprocess_dataset(df, image_dir)

    if len(train_ds) == 0:
        raise ValueError("No training samples were created. Check that your captions reference images present in the dataset.")

    inv_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dev_ds, batch_size=16)

    print(f"Vocab size: {vocab_size}")
    print(f"Train samples: {len(train_ds)}")

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)

    epoch_losses, epoch_bleus, epoch_times, epoch_list = [], [], [], []

    print("Starting Training...\n")

    for epoch in range(1, num_epochs + 1):
        start_epoch = time.time()
        encoder.train()
        decoder.train()
        running_loss = 0.0
        num_batches = 0

        for imgs, caps, txt in train_loader:
            imgs, caps = imgs.to(device), caps.to(device)
            feats = encoder(imgs)
            outputs = decoder(feats, caps[:, :-1])
            loss = criterion(outputs.reshape(-1, vocab_size), caps.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        epoch_time = time.time() - start_epoch
        avg_loss = running_loss / max(1, num_batches)

        if epoch == 1 or epoch % eval_every == 0 or epoch == num_epochs:
            bleu = evaluate_bleu(dev_loader, encoder, decoder, inv_vocab, device)

            epoch_losses.append(avg_loss)
            epoch_bleus.append(bleu)
            epoch_times.append(epoch_time)
            epoch_list.append(epoch)

            torch.save({
                'epoch': epoch,
                'encoder_state': encoder.state_dict(),
                'decoder_state': decoder.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss
            }, f'checkpoint_epoch_{epoch}.pth')

            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f} | BLEU: {bleu:.4f} | Time: {epoch_time:.2f}s")

    print("\n🎯 Training Complete")
    print(f"Total time (s): {sum(epoch_times):.2f}")

    best_epoch = epoch_list[epoch_bleus.index(max(epoch_bleus))]
    print(f"🏆 Best Epoch: {best_epoch} (BLEU: {max(epoch_bleus):.4f})")

    #Plotting
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epoch_list, epoch_losses, marker='o')
    plt.title('Loss per Epoch')

    plt.subplot(1, 2, 2)
    plt.plot(epoch_list, epoch_bleus, marker='o')
    plt.title('BLEU score per Epoch')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train()
