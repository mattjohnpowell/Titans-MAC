from datasets import load_dataset
from transformers import AutoTokenizer
from titans_model import TitansMAC
from titans_training import TextDataset, TitansTrainer
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
import datetime
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

torch.autograd.set_detect_anomaly(True)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def train_and_test_titans():
    # Set device
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    print(torch.cuda.get_device_name(0))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Load Dataset using Hugging Face datasets ---
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    # Create dataset and get tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(dataset, tokenizer, max_length=128)

    # --- Use the entire dataset ---
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Increased batch size

    # Initialize model (increased capacity)
    model = TitansMAC(
        dim=256,  # Increased embedding size
        num_heads=8,  # Increased number of heads
        memory_depth=4,  # Increased memory depth
        context_size=128
    )
    model.apply(init_weights)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Initialize trainer
    trainer = TitansTrainer(
        model,
        vocab_size=tokenizer.vocab_size,
        tokenizer=tokenizer,
        learning_rate=1e-5,  # You might need to adjust this
        device=device
    )
    trainer.embedding.apply(init_weights)
    trainer.output_layer.apply(init_weights)

    # Training loop
    print("\nStarting training...")
    num_epochs = 100  # Increased number of epochs
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        loss = trainer.train_epoch(dataloader)
        print(f"Epoch loss: {loss:.4f}")

        # Generate text periodically to monitor progress
        if (epoch + 1) % 5 == 0:
            print("\nGenerating text...")
            prompts = [
                "Artificial intelligence",
                "Machine learning is",
                "Deep learning",
                "Neural networks",
                "The future of AI"
            ]
            for prompt in prompts:
                generated = trainer.generate(
                    prompt,
                    max_length=100,
                    temperature=0.7
                )
                print(f"\nPrompt: {prompt}")
                print(f"Generated: {generated}")

    # Save model
    print("\nSaving model...")
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    model_filename = f"titans_model_{timestamp}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_state_dict': trainer.embedding.state_dict(),
        'output_layer_state_dict': trainer.output_layer.state_dict(),
    }, model_filename)
    print(f"Model saved as {model_filename}")

if __name__ == "__main__":
    train_and_test_titans()