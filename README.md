# Titan-MAC

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Titan-MAC is a PyTorch implementation of a novel neural network architecture inspired by the Titans model, incorporating a unique long-term memory mechanism. This allows the model to retain and utilize information over extended sequences, enhancing its ability to learn complex patterns and generate coherent text. The core innovation is the **Memory as Context (MAC)** module, which enables the network to dynamically query and integrate information from its learned memory into its current processing.

## Key Features

*   **Long-Term Memory:** A trainable memory module that stores and updates important information over time.
*   **Memory as Context (MAC):** A mechanism for dynamically integrating memory into the model's processing.
*   **Multi-Head Attention:** Enables the model to focus on different parts of the input and memory.
*   **Feedforward Network (FFN):** Enhances the model's ability to learn complex relationships.
*   **Training Script:** Includes a `titans_training.py` script for training the model on text datasets.
*   **Text Generation:** The trained model can generate coherent and contextually relevant text.

## Architecture

The model consists of the following key components:

1.  **Embedding Layer:** Converts input tokens into vector representations.
2.  **TitansMAC Layers:** A stack of layers, each containing:
    *   **Long-Term Memory:** Processes and updates the memory based on the current input.
    *   **Memory Query:** Generates a query vector to retrieve relevant information from the memory.
    *   **Multi-Head Attention:** Attends to the input, memory, and a set of persistent memory tokens.
    *   **Feedforward Network:** Further processes the attended information.
3.  **Output Layer:** Projects the final hidden state to the vocabulary space, producing probabilities for the next token.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/mattjohnpowell/Titans-MAC
    cd Titans-MAC
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

1.  **Prepare your dataset:** The training script expects a text dataset in a format compatible with the Hugging Face `datasets` library (e.g., Wikitext). You can modify the `train_and_test_titans` function in `titans_training.py` to use a different dataset.
2.  **Run the training script:**

    ```bash
    python titans_demo.py
    ```

    This will train the model and save checkpoints periodically.

### Text Generation

After training, you can generate text using the `generate` function in `titans_training.py`. You can modify the prompts in the main script to experiment with different starting points.

## Example

```python
from titans_training import TitansTrainer

# ... (load your trained model) ...

trainer = TitansTrainer(model, vocab_size, tokenizer)
trainer.load_state_dict(torch.load("path/to/your/model.pt")) # Load the trained model

prompt = "Artificial intelligence is"
generated_text = trainer.generate(prompt, max_length=100, temperature=0.7)
print(generated_text)