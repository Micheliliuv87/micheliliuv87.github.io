---
layout: distill
title: Autoencoder Basics and How to Implement
date: 2025-05-14 14:36:12
description: "Simple implementation of autoencoder framework"
tags: Notes
categories: "NeuralNetwork"
citation: true
authors:
  - name: Micheli Liu
    url: "https://micheliliuv87.github.io/"
    affiliations: 
        name: Emory University ISOM

toc:
  - name: Sparse Autoencoder (SAE)
giscus_comments: true

---
# (Still Updating)

## **Sparse Autoencoder (SAE)**

### **Definition:** 

A sparse autoencoder is a type of neural network used to learn compact, interpretable representations of data by enforcing sparsity in its hidden layer activations. Unlike regular autoencoders that mainly compress and reconstruct data, sparse autoencoders add a sparsity penalty (such as L1 regularization or KL divergence) to encourage only a small subset of neurons to be activated at any given time, leading to extraction of the most salient features and helping prevent overfitting. (Check out very interesting [lecture notes](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) by Andrew Ng)

### **Understanding in Simple Words & Implementation**

* “A SAE enforces sparsity in the hidden layer activations”

	*	Regular autoencoders minimize reconstruction error only.
	*	Sparse autoencoders add a constraint: they penalize the model when too many hidden units are active.
	*	This pushes the model to keep most neuron activations near zero.


* “The idea is to make most of the neurons inactive, forcing the model to learn efficient feature representations.”

	*	Focus on a few features at a time,
	*	Learn more interpretable and robust representations,
	*	Avoid overfitting or trivial identity mappings.

* **Loss Function**:
	$$\text{L} = \sum_{i=1}^{n} \|X_i - X_i{\prime}\|^2 + \lambda \sum |z_i|$$

**This loss function does two things:**

1.	Minimizes reconstruction error so that the autoencoder can accurately represent the input.

2.	Adds an L1 penalty on the hidden activations z_i, encouraging sparse activations.

This is similar to how Lasso regression enforces sparsity in coefficients.

<br>

| Symbol                     | Meaning                                                              |
|----------------------------|----------------------------------------------------------------------|
| $X_i$                  | The original input sample                                            |
| $X_i'$                 | The reconstructed output (decoded from latent representation)       |
| $\|X_i - X_i'\|^2$     | Reconstruction loss: how close the output is to the input            |
| $z_i$                  | Hidden layer activation for input \( i \)                            |
| $\sum \lvert z_i \rvert$            | Sum of absolute hidden activations — the sparsity penalty            |
| $\lambda$              | Regularization parameter controlling the strength of sparsity        |

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 1) Define the Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        # encoder: input → hidden
        self.encoder = nn.Linear(input_dim, hidden_dim)
        # decoder: hidden → reconstruction
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor):
        # encode with sigmoid activation
        z = torch.sigmoid(self.encoder(x))
        # decode with sigmoid (or no activation, depending on your data)
        x_recon = torch.sigmoid(self.decoder(z))
        return x_recon, z

# 2) Hyperparameters
input_dim    = 784   # e.g. flattened 28×28 image
hidden_dim   = 64    # size of latent (bottleneck)
lambda_sparse = 1e-3 # weight of the sparsity penalty
learning_rate = 1e-2

# 3) Instantiate model, loss & optimizer
model = SparseAutoencoder(input_dim, hidden_dim)
mse_loss = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 4) Dummy data loader (replace with real dataset)
#    Here: 100 samples of dimension 784
data = torch.randn(100, input_dim)

# 5) Training loop (one epoch for illustration)
model.train()
for x in data:            # if you have a real DataLoader, iterate batches
    x = x.unsqueeze(0)    # make it shape (1, 784)
    
    # Forward pass
    x_recon, z = model(x)
    
    # Reconstruction loss
    recon_loss = mse_loss(x_recon, x)
    
    # Sparsity loss: L1 on hidden activations
    # Sum over hidden dims, mean over batch
    sparsity_loss = torch.mean(torch.abs(z))
    
    # Total loss
    loss = recon_loss + lambda_sparse * sparsity_loss
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Recon: {recon_loss.item():.4f}  |  Sparse: {sparsity_loss.item():.4f}  |  Total: {loss.item():.4f}")
```
```md
Recon: 1.2073  |  Sparse: 0.5127  |  Total: 1.2078
Recon: 1.3084  |  Sparse: 0.5057  |  Total: 1.3089
Recon: 1.1943  |  Sparse: 0.5150  |  Total: 1.1949
Recon: 1.3257  |  Sparse: 0.4914  |  Total: 1.3262
Recon: 1.1187  |  Sparse: 0.4987  |  Total: 1.1192
Recon: 1.0752  |  Sparse: 0.4697  |  Total: 1.0756
Recon: 1.1239  |  Sparse: 0.5137  |  Total: 1.1244
...
```


---

## **References**

1. [Sparse Autoencoder Lecture Notes by Andrew Ng](https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf) 
