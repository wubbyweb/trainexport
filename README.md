# Transformer Intent Classifier 🧠

This project demonstrates a **Character-level Transformer** built from scratch using pure **NumPy**. It is designed to classify user queries into three distinct intents for a RAG-based chat application.

## 🚀 Overview

The model is a minimalist implementation of the Transformer architecture, focusing on the core mathematical operations (Attention, MLP, Backpropagation) without using heavy frameworks like PyTorch or TensorFlow.

### Key Features
- **Pure NumPy Implementation:** Understand the forward and backward pass from first principles.
- **Character-level Tokenization:** No complex word vocabularies—every character is a token.
- **Intent Classification:** Classifies messages into:
  - `S` (Semantic query)
  - `D` (Date-specific query)
  - `R` (Recency query)
- **Weight Export:** Trains and saves weights to a lightweight `intent_weights.json` for edge deployment.

## 📁 Project Structure

```text
.
├── source/                 # Original Python source code
│   └── trainexport.py      # Core model logic and training script
├── tutorial_trainexport.ipynb  # Step-by-step Jupyter tutorial (Colab compatible)
├── generate_notebook.py    # Utility script to generate the notebook from source
└── README.md               # You are here!
```

## 🛠️ Getting Started

### 1. Training the Model
Open the `tutorial_trainexport.ipynb` notebook and run all cells. This will:
- Generate a synthetic dataset.
- Initialize the model weights.
- Run the training loop for 5,000 steps.
- Export `intent_weights.json`.

Alternatively, you can run the script directly:
```bash
python source/trainexport.py
```

### 2. Loading the Model
The exported `intent_weights.json` contains:
- Model hyperparameters (`n_embd`, `n_head`, etc.)
- The tokenizer's unique character mapping (`uchars`)
- The learned weight matrices for all layers.

## 🧠 Model Architecture

The model follows the standard Decoder-only Transformer architecture (like GPT):
- **Embedding Layer:** Token and Position embeddings.
- **Attention Blocks:** Multi-head self-attention with causal masking.
- **Feed-Forward Layers:** ReLU-activated MLPs.
- **Optimization:** Custom Adam optimizer implemented with NumPy.

## 📝 License
MIT
