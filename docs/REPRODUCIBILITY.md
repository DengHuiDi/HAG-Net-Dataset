# Reproducibility

- Image size: 640
- Classes: class 0 `fire`, class 1 `smoke`
- Dataset A main runs: seeds 0, 1, and 2
- Scale AP: COCO bbox, IoU 0.50:0.95, maxDets=100, original-image box area
- Scales: small < 32², medium 32² to < 96², large >= 96² pixels
- Background false-alarm operating point: confidence 0.25, NMS IoU 0.7

`weights/HAG-Net.torchscript` is the portable image-size-640 inference artifact.
