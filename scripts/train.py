"""Train HAG-Net with a selected dataset YAML."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.build import build_hagnet

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/dataset_a.yaml")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build_hagnet().train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device, seed=a.seed)

if __name__ == "__main__":
    main()
