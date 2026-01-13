#!/usr/bin/env python3
"""
router_activation_extractor.py
Extracts routing patterns from MiniMax M2.1 inference for VQ-VAE training.

Author: Claude Big Dog Opus 4.5
Date: 2026-01-12
Part of: Sviluppo Research / Rocketeer Project
"""

import torch
from typing import List, Dict, Optional
import json
from pathlib import Path
import numpy as np


class RouterActivationHook:
    """Hook to capture router activations during inference."""
    
    def __init__(self, layer_indices: List[int]):
        self.layer_indices = layer_indices
        self.activations: Dict[int, torch.Tensor] = {}
        self.handles: List = []
    
    def hook_fn(self, layer_idx: int):
        def fn(module, input, output):
            # Extract router logits - architecture dependent
            if hasattr(output, 'router_logits'):
                router_logits = output.router_logits
            elif isinstance(output, tuple) and len(output) > 1:
                router_logits = output[1]
            else:
                router_logits = None
            
            if router_logits is not None:
                self.activations[layer_idx] = router_logits.detach().cpu()
        return fn
    
    def attach(self, model):
        """Attach hooks to specified layers."""
        for layer_idx in self.layer_indices:
            try:
                layer = model.model.layers[layer_idx]
                # Try different MoE attribute names
                moe_module = None
                for attr in ['block_sparse_moe', 'moe', 'sparse_moe']:
                    if hasattr(layer, attr):
                        moe_module = getattr(layer, attr)
                        break
                
                if moe_module and hasattr(moe_module, 'gate'):
                    handle = moe_module.gate.register_forward_hook(
                        self.hook_fn(layer_idx)
                    )
                    self.handles.append(handle)
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not attach hook to layer {layer_idx}: {e}")
    
    def detach(self):
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
    
    def get_routing_trajectory(self) -> Optional[torch.Tensor]:
        """Return stacked routing activations across layers."""
        if not self.activations:
            return None
        layers = sorted(self.activations.keys())
        return torch.stack([self.activations[l] for l in layers], dim=0)
    
    def reset(self):
        """Clear stored activations."""
        self.activations = {}


def extract_trajectories(
    model,
    tokenizer,
    prompts: List[str],
    target_layers: Optional[List[int]] = None,
    max_length: int = 512
) -> List[torch.Tensor]:
    """
    Extract routing trajectories for a batch of prompts.
    
    Args:
        model: HuggingFace model with MoE architecture
        tokenizer: Corresponding tokenizer
        prompts: List of input prompts
        target_layers: Which layers to capture (default: 40-60)
        max_length: Maximum sequence length
    
    Returns:
        List of tensors, each shape [num_layers, seq_len, num_experts]
    """
    if target_layers is None:
        target_layers = list(range(40, 60))
    
    hook = RouterActivationHook(target_layers)
    hook.attach(model)
    
    trajectories = []
    
    for i, prompt in enumerate(prompts):
        hook.reset()
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        ).to(model.device)
        
        with torch.no_grad():
            _ = model(**inputs)
        
        trajectory = hook.get_routing_trajectory()
        if trajectory is not None:
            trajectories.append(trajectory)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(prompts)} prompts")
    
    hook.detach()
    return trajectories


def save_trajectory_dataset(
    trajectories: List[torch.Tensor],
    metadata: List[dict],
    output_path: str
):
    """Save trajectories in format for VQ-VAE training."""
    # Convert to numpy for JSON serialization
    data = {
        'trajectories': [t.numpy().tolist() for t in trajectories],
        'metadata': metadata,
        'num_trajectories': len(trajectories),
        'trajectory_shape': list(trajectories[0].shape) if trajectories else None
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f)
    
    print(f"Saved {len(trajectories)} trajectories to {output_path}")
    if trajectories:
        print(f"Trajectory shape: {trajectories[0].shape}")


def load_trajectory_dataset(input_path: str) -> tuple:
    """Load saved trajectories."""
    with open(input_path) as f:
        data = json.load(f)
    
    trajectories = [torch.tensor(t) for t in data['trajectories']]
    metadata = data['metadata']
    
    return trajectories, metadata


# Mock extraction for testing without model
def create_mock_trajectories(
    num_trajectories: int = 100,
    num_layers: int = 20,
    seq_len: int = 64,
    num_experts: int = 8
) -> List[torch.Tensor]:
    """Create synthetic trajectories for testing."""
    trajectories = []
    for _ in range(num_trajectories):
        # Simulate sparse expert selection with softmax
        logits = torch.randn(num_layers, seq_len, num_experts)
        # Add some temporal structure
        for l in range(1, num_layers):
            logits[l] = 0.7 * logits[l] + 0.3 * logits[l-1]
        trajectories.append(logits)
    return trajectories


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract router activations")
    parser.add_argument("--output", default="./routing_trajectories.json")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--num-mock", type=int, default=100)
    args = parser.parse_args()
    
    if args.mock:
        print("Generating mock trajectories...")
        trajectories = create_mock_trajectories(num_trajectories=args.num_mock)
        metadata = [{"prompt": f"mock_prompt_{i}"} for i in range(len(trajectories))]
    else:
        # Real extraction requires model
        print("Real extraction requires --model-path argument")
        print("Using mock data instead...")
        trajectories = create_mock_trajectories(num_trajectories=args.num_mock)
        metadata = [{"prompt": f"mock_prompt_{i}"} for i in range(len(trajectories))]
    
    save_trajectory_dataset(trajectories, metadata, args.output)
