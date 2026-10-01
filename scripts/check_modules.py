"""Run lightweight forward checks for the three HAG-Net modules."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.modules import CAFM, DMMM, GradientAwareDecoupledDetectionHead

def main():
    x = torch.randn(1, 64, 32, 32)
    assert DMMM(64)(x).shape == x.shape
    assert CAFM(64)(x).shape == x.shape
    head = GradientAwareDecoupledDetectionHead(nc=2, hidc=64, ch=(64, 128, 256))
    features = [torch.randn(1, 64, 20, 20), torch.randn(1, 128, 10, 10), torch.randn(1, 256, 5, 5)]
    head.train()
    assert [tuple(y.shape) for y in head(features)] == [(1, 66, 20, 20), (1, 66, 10, 10), (1, 66, 5, 5)]
    print("All HAG-Net module checks passed.")

if __name__ == "__main__":
    main()
