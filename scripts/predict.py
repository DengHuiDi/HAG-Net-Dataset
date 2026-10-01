"""Run HAG-Net inference with the portable TorchScript model."""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--weights", default="weights/HAG-Net.torchscript")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", default="")
    a = p.parse_args()
    YOLO(a.weights).predict(source=a.source, imgsz=a.imgsz, conf=a.conf, device=a.device, save=True)

if __name__ == "__main__":
    main()
