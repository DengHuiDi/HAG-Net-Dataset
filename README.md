# HAG-Net

Official resources for HAG-Net, a compact detector for RGB UAV fire and smoke detection.

## Contents

- `models/`: model code and implementations of DMMM, CAFM, and the Gradient-aware Decoupled Detection Head.
- `weights/`: HAG-Net and YOLOv11n baseline checkpoints, plus a portable HAG-Net TorchScript artifact.
- `configs/`: dataset configuration templates.
- `splits/dataset_a/`: the fixed Dataset A train/validation image-ID and checksum manifest (no images or labels).
- `splits/fsd1902/`: fixed FSD-1902 train/validation/test lists.
- `annotations/fsd1902/`: FSD-1902 image-name mapping and annotation-package instructions.
- `scripts/`: training, validation, prediction, and module verification entry points.
- `results/`: verified experimental summaries.
- [`annotations/fsd1902/FSD-1902-annotations-only.zip`](annotations/fsd1902/FSD-1902-annotations-only.zip): team-added YOLO labels, fixed lists, mapping, and instructions; no source images or videos.

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

## Dataset A

The main experiments use a fixed split of 4,734 training and 1,639 validation images, with class 0 = `fire` and class 1 = `smoke`. The [Dataset A split manifest](splits/dataset_a/dataset_a_manifest.csv) publishes image IDs, relative image and label paths, split membership, SHA-256 checksums, and qualified source categories. Dataset A images and label files are not redistributed here. Source categories follow the documented dataset-organisation rules and are marked `probable`, not independently verified original-file provenance. See the [manifest notes](splits/dataset_a/README.md) for field definitions and distribution scope.

FSD-1902 uses the fixed 1,332/380/190 lists in `splits/fsd1902/`. Its source RGB imagery comes from Hopkins et al.'s FLAME 2 dataset ([IEEE DataPort DOI: 10.21227/swyw-6j78](https://doi.org/10.21227/swyw-6j78)) and must be obtained from that provider under its applicable terms. This repository distributes only the team's added fire/smoke annotations, fixed split, and [image-name mapping/status](annotations/fsd1902/image_mapping.jsonl), not FLAME 2 source images. The source-to-working-image rename mapping has not been recovered, so the official download cannot be paired automatically with every label; see the [assembly instructions](annotations/fsd1902/README.md). D-Fire must be obtained separately from its original provider. See `docs/DATASETS.md` for further dataset information.

## Checkpoints and model files

| Model | Checkpoint | SHA-256 of original checkpoint | Repository model file |
| --- | --- | --- | --- |
| YOLOv11n, seed 0 | [`weights/YOLOv11n_baseline.pt`](weights/YOLOv11n_baseline.pt) | `43442d4e69132f565bd032d63f9940b9d2bf932bafe0e65d552744576e21a8ca` | Upstream Ultralytics `yolo11.yaml` |
| HAG-Net, seed 0 | [`weights/HAG-Net.pt`](weights/HAG-Net.pt) | `5faa2ec91eb7f23092fb9cb91789d5acbfd5366989dd9e0c8962a8d9020a0be8` | [`models/HAG-Net.yaml`](models/HAG-Net.yaml) |

The checkpoint files are tracked with Git LFS. Their LFS object IDs match the SHA-256 values of the original local `best.pt` files; binary downloads from a clean environment have not yet been independently tested. `weights/HAG-Net.torchscript` is a separate inference artifact, not a substitute for the seed-0 checkpoint.

See `docs/RESULTS.md` and `docs/REPRODUCIBILITY.md` for additional details.

## License

The code follows the repository license. FLAME 2 source images remain under the original provider's copyright and use conditions; the code license does not apply to or grant rights in those images. The archived DataPort page lists “Creative Commons Attribution,” but the license version and complete legal-text link have not been verified from the original download package. External datasets remain subject to their respective licenses and citation requirements.
