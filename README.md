# HAG-Net

Official resources for HAG-Net, a compact detector for RGB UAV fire and smoke detection.

## Contents

- `models/`: HAG-Net configuration and the DMMM, CAFM, and Gradient-aware Decoupled Detection Head implementations.
- `weights/`: final HAG-Net checkpoint, portable TorchScript model, and YOLOv11n baseline checkpoint.
- `configs/`: dataset configuration templates.
- `splits/fsd1902/`: fixed FSD-1902 train/validation/test lists.
- `scripts/`: training, validation, prediction, and module verification entry points.
- `results/`: verified experimental summaries.
- `FSD-1902.zip`: FSD-1902 RGB images and YOLO annotations.

## Installation

```bash
pip install -r requirements.txt
```

## Quick inference

```bash
python scripts/predict.py --source path/to/image_or_video
```

## Build check

```bash
python scripts/check_modules.py
python models/build.py
```

Dataset classes are class 0 `fire` and class 1 `smoke`. Dataset A images are not redistributed. FSD-1902 uses the fixed 1,332/380/190 lists supplied in `splits/fsd1902/`. D-Fire must be obtained separately from its official source.

See `docs/DATASETS.md`, `docs/RESULTS.md`, and `docs/REPRODUCIBILITY.md` for details.

## License

The code follows the repository license. External datasets remain subject to their respective licenses and citation requirements.
