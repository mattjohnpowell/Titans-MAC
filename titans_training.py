import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from titans_model import TitansMAC
import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for example in tqdm(hf_dataset):
            text = example["text"]
            if len(text) > 0:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=self.max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                self.examples.append(encoding['input_ids'].squeeze(0))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        encoding = self.examples[idx]
        x = encoding
        y = encoding.clone()
        return x, y

class TitansTrainer:
    def __init__(
        self,
        model,
        vocab_size,
        tokenizer=None,
        learning_rate=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model moved to {device}")
        
        # Add embedding and output layers
        self.embedding = nn.Embedding(vocab_size, model.dim).to(device)
        self.output_layer = nn.Linear(model.dim, vocab_size).to(device)
        
        # Initialize optimizer
        self.optimizer = AdamW([
            {'params': model.parameters()},
            {'params': self.embedding.parameters()},
            {'params': self.output_layer.parameters()}
        ], lr=learning_rate)

        # Add a learning rate scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=2, factor=0.5, verbose=True)

        
        self.prev_memory = None
        self.prev_surprise = None

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0



        for batch_idx, (x, y) in enumerate(tqdm(dataloader)):
            x, y = x.to(self.device), y.to(self.device)

            # Shift the target to the right for language modeling
            y_input = y[:, :-1]  # Input for the decoder (remove the last token)
            y_expected = y[:, 1:]  # Expected output (remove the first token)

            # Forward pass
            embedded = self.embedding(y_input) # Use y_input here
            output, _, _ = self.model(
                embedded,
                None,
                None
            )

            # Crop output to match y_expected
            output = output[:, :y_expected.size(1), :] # Crop the output sequence

            # Compute loss
            logits = self.output_layer(output)
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), # Reshape logits to (batch_size * sequence_length, vocab_size)
                y_expected.contiguous().view(-1),  # Flatten y_expected to (batch_size * sequence_length)
                ignore_index=self.tokenizer.pad_token_id
            )

            # Backward pass
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.05)  # Stricter clipping
            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}')

        self.scheduler.step(total_loss)
        return total_loss / len(dataloader)

    def generate(self, prompt, max_length=100, temperature=0.7):
        self.model.eval()
        tokenizer = self.tokenizer

        # Tokenize prompt
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(self.device)

        # Initialize states
        prev_memory = None
        prev_surprise = None

        # Generate tokens
        for _ in range(max_length):
            # Get embeddings for the entire sequence so far
            embedded = self.embedding(input_ids)
            print(f"Step: {_}, embedded min: {torch.min(embedded)}, embedded max: {torch.max(embedded)}, any NaN: {torch.isnan(embedded).any()}, any Inf: {torch.isinf(embedded).any()}")

            # Inspect prev_memory and prev_surprise
            print(f"Step: {_}, prev_memory min: {torch.min(prev_memory) if prev_memory is not None else None}, prev_memory max: {torch.max(prev_memory) if prev_memory is not None else None}, any NaN: {torch.isnan(prev_memory).any() if prev_memory is not None else False}, any Inf: {torch.isinf(prev_memory).any() if prev_memory is not None else False}")
            print(f"Step: {_}, prev_surprise min: {torch.min(prev_surprise) if prev_surprise is not None else None}, prev_surprise max: {torch.max(prev_surprise) if prev_surprise is not None else None}, any NaN: {torch.isnan(prev_surprise).any() if prev_surprise is not None else False}, any Inf: {torch.isinf(prev_surprise).any() if prev_surprise is not None else False}")


            # Ensure prev_memory has the same sequence length as embedded
            if prev_memory is not None:
                # Pad or truncate prev_memory along the sequence dimension
                if prev_memory.size(1) < embedded.size(1):
                    padding = torch.zeros(
                        (embedded.size(0), embedded.size(1) - prev_memory.size(1), embedded.size(2)),
                        dtype=prev_memory.dtype,
                        device=prev_memory.device
                    )
                    prev_memory = torch.cat([prev_memory, padding], dim=1)
                    prev_surprise = torch.cat([prev_surprise, padding], dim=1)
                elif prev_memory.size(1) > embedded.size(1):
                    prev_memory = prev_memory[:, :embedded.size(1), :]
                    prev_surprise = prev_surprise[:, :embedded.size(1), :]

            # Forward pass with the full sequence
            with torch.no_grad():
                output, prev_memory, prev_surprise = self.model(
                    embedded,
                    prev_memory,
                    prev_surprise
                )
            
            # Get next token probabilities from the last position
            # Inspect output of the model
            print(f"Step: {_}, model output min: {torch.min(output)}, model output max: {torch.max(output)}, any NaN: {torch.isnan(output).any()}, any Inf: {torch.isinf(output).any()}")

            logits = self.output_layer(output[:, -1])
            # Inspect output[:, -1] and output_layer weights
            print(f"Step: {_}, output[:, -1] min: {torch.min(output[:, -1])}, output[:, -1] max: {torch.max(output[:, -1])}, any NaN: {torch.isnan(output[:, -1]).any()}, any Inf: {torch.isinf(output[:, -1]).any()}")
            print(f"Step: {_}, output_layer weights min: {torch.min(self.output_layer.weight)}, output_layer weights max: {torch.max(self.output_layer.weight)}, any NaN: {torch.isnan(self.output_layer.weight).any()}, any Inf: {torch.isinf(self.output_layer.weight).any()}")

            print(f"Step: {_}, logits min: {torch.min(logits)}, logits max: {torch.max(logits)}, any NaN: {torch.isnan(logits).any()}, any Inf: {torch.isinf(logits).any()}")

            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            print(f"Step: {_}, probs min: {torch.min(probs)}, probs max: {torch.max(probs)}, any NaN: {torch.isnan(probs).any()}, any Inf: {torch.isinf(probs).any()}")
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Append to input
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if end of text token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        # Decode generated text
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text
