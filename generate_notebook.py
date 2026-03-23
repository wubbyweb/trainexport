import os
import json
import re

source_path = '/Users/rj/Programs/trainexport/source/trainexport.py'
notebook_path = '/Users/rj/Programs/trainexport/tutorial_trainexport.ipynb'

with open(source_path, 'r', encoding='utf-8') as f:
    text = f.read()

parts = re.split(r'# ── (.*?) ─+', text)

def markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in content.strip().split("\n")]
    }

def code_cell(content):
    lines = content.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in lines[:-1]] + [lines[-1]] if lines else []
    }

cells = []

# Title
cells.append(markdown_cell(
'''# Transformer Intent Classification Tutorial
Welcome to this detailed, step-by-step tutorial on building and training a Character-level Transformer from scratch using pure Numpy!

This notebook takes the `trainexport.py` source code and breaks it down cell by cell. Before each code block, we explain what the code does and the intuition behind it. 
It operates by classifying sentences into three intents:
1. `S` (Semantic query)
2. `D` (Date-specific query)
3. `R` (Recency query)
'''
))

# Top part (imports)
cells.append(markdown_cell(
'''## 1. Imports & Setup
We start by importing the required standard libraries and setting up our environment. 
- `math` and `np` (NumPy) will be used for computations.
- `random` will be used to shuffle the dataset.

We also set a fixed seed (`42`) for both Python's random and Numpy so that we get reproducible results every time we run this notebook.'''
))
cells.append(code_cell(parts[0]))

descriptions = {
    'DATASET': '''## 2. Dataset Generation
Here we define our synthetic dataset `raw_docs`. Each string starts with a label character `S`, `D`, or `R` followed by a pipe `|` and then the chat message text.
We shuffle the dataset in-place using `random.shuffle()` to ensure our stochastic gradient descent sees varied examples and doesn't get stuck.''',
    
    'TOKENIZER': '''## 3. Character-level Tokenizer
A Transformer operates on numbers, not text. For this tutorial, we will use a character-level tokenizer. 
We collect all unique characters found in our training corpus (`uchars`). We then assign a unique integer ID to every token. 

- `vocab_size` is the total number of unique characters, plus 1 for the End-of-Sequence / Beginning-of-Sequence (BOS) token.
- `tokenize(doc)` converts a text string into a list of integers, adding `BOS` at the beginning and the end.''',

    'HYPERPARAMS': '''## 4. Hyperparameters Configuration
We define the structural dimensions of our Transformer model:
- `n_embd`: The embedding dimension, the size of the vectors propagating through the model (`16`).
- `n_head`: The number of attention heads (`4`).
- `n_layer`: The number of transformer blocks (`2`).
- `block_size`: The maximum context window size length (`64`).
- `head_dim`: Embedding dimension per attention head (`n_embd // n_head` = 4).''',
    
    'WEIGHTS': '''## 5. Parameter Initialization
Here, we initialize all the learnable weights of the model using a normal distribution via `np.random.randn`. We scale the initial random weights by `0.08` for stability.

We declare a unified dictionary `W` containing:
- Token and Position embeddings (`wte`, `wpe`)
- Query, Key, Value, and Output projection matrices for each layer (`wq`, `wk`, `wv`, `wo`)
- Feedforward layer weights (`fc1`, `fc2`)
- Final generation / Language Model head (`lm_head`)''',
    
    'FORWARD': '''## 6. Forward Pass & Loss Functions
This section implements the core mathematical operations for inference and calculating gradients.
1. `softmax_np(x)` and `rmsnorm(x)`: Root Mean Square Layer Normalization and Softmax stabilize neural network activations and output valid probability masks.
2. `forward(tokens)`: The main Transformer forward pass which executes the calculations. It performs Embedding lookup, Multi-Head Self Attention (computes $QK^T$ scaled by $\sqrt{D}$, applies causal masking using `np.triu`, and multiplies by $V$), and Feed-Forward Networks using ReLU. It saves intermediate results into `cache`.
3. `loss_and_grad(...)`: Computes Cross-Entropy loss over the vocabulary predictions and the initial gradient.
4. `backward(...)`: Computes full backpropagation using gradients from `dl` to manually trace partial derivatives down the network into dictionary `G`.''',

    'ADAM': '''## 7. Adam Optimizer & Training Loop
With our forward and backward functions complete, we implement a custom Adam optimizer to train our custom network!
- `mA` and `vA` track the first and second moments of the gradients.
- We loop through 5000 steps (`num_steps`). In every step, we fetch a document, compute forward, calculate loss, compute backward gradients `G`, and apply Adam updates to every weight in `W` towards optimal minima.''',

    'EXPORT': '''## 8. Export Model Weights
After our model converges, we package its knowledge into a standard format. We serialize the configuration hyperparameters, the tokenizer characters, the intention labels, and every learned weight matrix (converted from numpy arrays into python lists).

These are exported out as `intent_weights.json`. This format allows lightweight platforms (like a web browser or a tiny embedded device) to load and execute models originally trained in full Python scripts!''',

    'SANITY CHECK': '''## 9. Evaluation & Sanity Check
Finally, we write a small script to evaluate our trained model against an unseen test dataset.
- `score(doc_str)` computes the average negative log-likelihood loss for a particular snippet.
- `classify(q)` evaluates the input query `q` against all three label prefixes `S|`, `D|`, and `R|`. The class that leads to the lowest perplexity (minimal loss) is returned as our final intent prediction!'''
}

for i in range(1, len(parts), 2):
    key = parts[i].strip()
    code = parts[i+1]
    
    # We find the description
    desc = descriptions.get(key, f"## Section: {key}")
    
    cells.append(markdown_cell(desc))
    cells.append(code_cell(code))

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {
            "name": "Transformer_Intent_Classifier_Tutorial.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print(f"Jupyter notebook saved to: {notebook_path}")
