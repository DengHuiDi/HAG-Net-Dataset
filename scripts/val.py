"""Validate the portable HAG-Net model."""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="configs/dataset_a.yaml")
    p.add_argument("--weights", default="weights/HAG-Net.torchscript")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0")
    a = p.parse_args()
    YOLO(a.weights).val(data=a.data, imgsz=a.imgsz, batch=a.batch, device=a.device)

if __name__ == "__main__":
    main()
