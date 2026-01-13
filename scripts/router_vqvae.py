#!/usr/bin/env python3
"""
router_vqvae.py
VQ-VAE adapted for routing trajectory discretization.

Based on: Rishav et al. (2025) "Behaviour Discovery and Attribution for Explainable RL"
Adapted by: Claude Big Dog Opus 4.5
Part of: Sviluppo Research / Rocketeer Project
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
import numpy as np
import json


class VectorQuantizer(nn.Module):
    """Vector quantization layer with codebook."""
    
    def __init__(self, num_codes: int, code_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_z = z.view(-1, self.code_dim)
        
        distances = (
            flat_z.pow(2).sum(1, keepdim=True)
            - 2 * flat_z @ self.codebook.weight.t()
            + self.codebook.weight.pow(2).sum(1)
        )
        
        indices = distances.argmin(dim=1)
        quantized_flat = self.codebook(indices)
        quantized = quantized_flat.view_as(z)
        indices = indices.view(z.shape[0], z.shape[1])
        
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        quantized = z + (quantized - z).detach()
        
        return quantized, vq_loss, indices


class RouterVQVAE(nn.Module):
    """VQ-VAE for routing trajectory discretization."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        code_dim: int = 64,
        num_codes: int = 128,
        num_layers: int = 4,
        nhead: int = 8
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.pre_vq = nn.Linear(hidden_dim, code_dim)
        self.vq = VectorQuantizer(num_codes, code_dim)
        self.post_vq = nn.Linear(code_dim, hidden_dim)
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        device = x.device
        
        h = self.input_proj(x)
        mask = self._generate_causal_mask(seq_len, device)
        h = self.encoder(h, mask=mask)
        
        z = self.pre_vq(h)
        quantized, vq_loss, indices = self.vq(z)
        h = self.post_vq(quantized)
        
        h = self.decoder(h, mask=mask)
        reconstruction = self.output_proj(h[:, :-1, :])
        
        return reconstruction, vq_loss, indices
    
    def get_codebook_vectors(self) -> np.ndarray:
        return self.vq.codebook.weight.detach().cpu().numpy()
    
    def encode_trajectory(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            h = self.input_proj(x)
            mask = self._generate_causal_mask(x.shape[1], x.device)
            h = self.encoder(h, mask=mask)
            z = self.pre_vq(h)
            _, _, indices = self.vq(z)
        return indices


class RoutingTrajectoryDataset(Dataset):
    """Dataset for routing trajectories."""
    
    def __init__(self, trajectories: List[np.ndarray]):
        self.trajectories = [torch.tensor(t, dtype=torch.float32) for t in trajectories]
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        return {"trajectory": self.trajectories[idx]}


def train_vqvae(
    model: RouterVQVAE,
    train_loader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    alpha: float = 1.0
) -> RouterVQVAE:
    """Train the Router-VQ-VAE."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        total_recon_loss = 0
        total_vq_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            x = batch['trajectory'].to(device)
            target = x[:, 1:, :]
            
            reconstruction, vq_loss, _ = model(x)
            
            recon_loss = F.mse_loss(reconstruction, target)
            loss = recon_loss + alpha * vq_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            num_batches += 1
        
        print(f"Epoch {epoch}: Recon={total_recon_loss/num_batches:.4f}, VQ={total_vq_loss/num_batches:.4f}")
    
    return model


def cluster_codebook(codebook: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """Apply spectral clustering to codebook vectors."""
    from sklearn.cluster import SpectralClustering
    
    clustering = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=10
    )
    labels = clustering.fit_predict(codebook)
    return labels


if __name__ == "__main__":
    # Demo with synthetic data
    num_experts = 8
    seq_len = 64
    batch_size = 16
    
    # Generate synthetic routing trajectories
    trajectories = [np.random.randn(seq_len, num_experts) for _ in range(100)]
    dataset = RoutingTrajectoryDataset(trajectories)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train model
    model = RouterVQVAE(input_dim=num_experts)
    model = train_vqvae(model, loader, num_epochs=5)
    
    # Get codebook and cluster
    codebook = model.get_codebook_vectors()
    print(f"Codebook shape: {codebook.shape}")
    
    labels = cluster_codebook(codebook, n_clusters=5)
    print(f"Cluster assignments: {np.bincount(labels)}")
