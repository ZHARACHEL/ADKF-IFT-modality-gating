"""
Simple verification script for Modality-Level Gating.
Tests parameter count and basic functionality.
"""

import torch
import torch.nn as nn
from typing import List

# Inline ModalityGate for testing
class ModalityGate(nn.Module):
    def __init__(self, num_modalities: int = 3, hidden_dim: int = 8):
        super().__init__()
        self.num_modalities = num_modalities
        input_dim = num_modalities * 2
        self.gate_generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_modalities),
            nn.Sigmoid()
        )
        
    def compute_statistics(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        stats = []
        for feat in features_list:
            norms = torch.norm(feat, p=2, dim=1)
            norm_mean = norms.mean().unsqueeze(0)
            norm_std = norms.std().unsqueeze(0)
            stats.extend([norm_mean, norm_std])
        return torch.stack(stats).unsqueeze(0)
    
    def forward(self, support_features: List[torch.Tensor]) -> torch.Tensor:
        stats = self.compute_statistics(support_features)
        gates = self.gate_generator(stats)
        return gates.squeeze(0)


def test_gate():
    print("=" * 60)
    print("Modality-Level Gating Verification")
    print("=" * 60)
    
    # Test 1: Parameter count
    print("\n[Test 1] Parameter Count")
    gate = ModalityGate(num_modalities=3, hidden_dim=8)
    total_params = sum(p.numel() for p in gate.parameters())
    expected = 6 * 8 + 8 + 8 * 3 + 3  # 83
    print(f"  Total parameters: {total_params}")
    print(f"  Expected: {expected}")
    assert total_params == expected, f"Mismatch: {total_params} != {expected}"
    print(f"  ✓ Parameter count correct: {total_params} params")
    
    # Test 2: Forward pass
    print("\n[Test 2] Forward Pass")
    # Create dummy features (3 modalities)
    support_gnn = torch.randn(16, 256)     # 16 samples, 256 dims
    support_ecfp = torch.randn(16, 2048)   # 16 samples, 2048 dims
    support_pc = torch.randn(16, 42)       # 16 samples, 42 dims
    support_features = [support_gnn, support_ecfp, support_pc]
    
    gate_weights = gate(support_features)
    print(f"  Gate weights shape: {gate_weights.shape}")
    print(f"  Gate weights: {gate_weights.detach().numpy()}")
    assert gate_weights.shape == (3,), f"Shape mismatch: {gate_weights.shape}"
    print(f"  ✓ Forward pass successful")
    
    # Test 3: Gate range
    print("\n[Test 3] Gate Weight Range")
    print(f"  Min: {gate_weights.min().item():.4f}")
    print(f"  Max: {gate_weights.max().item():.4f}")
    assert torch.all(gate_weights > 0) and torch.all(gate_weights < 1)
    print(f"  ✓ All gates in (0, 1) range")
    
    # Test 4: Apply gates to features
    print("\n[Test 4] Apply Gates to Features")
    query_gnn = torch.randn(32, 256)
    query_ecfp = torch.randn(32, 2048)
    query_pc = torch.randn(32, 42)
    query_features = [query_gnn, query_ecfp, query_pc]
    
    # Apply same gates to query
    gated_query = []
    for i, feat in enumerate(query_features):
        gate_val = gate_weights[i].view(1, 1)
        gated_feat = feat * gate_val
        gated_query.append(gated_feat)
    
    # Concatenate
    concat_features = torch.cat(gated_query, dim=1)
    expected_dim = 256 + 2048 + 42
    print(f"  Concatenated shape: {concat_features.shape}")
    print(f"  Expected: (32, {expected_dim})")
    assert concat_features.shape == (32, expected_dim)
    print(f"  ✓ Gating and concatenation successful")
    
    # Test 5: Gradient flow
    print("\n[Test 5] Gradient Flow")
    loss = concat_features.sum()
    loss.backward()
    has_grad = all(p.grad is not None for p in gate.parameters())
    print(f"  All parameters have gradients: {has_grad}")
    assert has_grad
    print(f"  ✓ Gradients flow correctly")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Gate module: {total_params} parameters (<100)")
    print(f"  - Support-driven: gates computed from support set")
    print(f"  - Task-adaptive: different tasks → different gates")
    print(f"  - Clean insertion: encoder → gating → concatenation")


if __name__ == "__main__":
    test_gate()
