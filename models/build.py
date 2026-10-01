"""Register HAG-Net modules with Ultralytics and construct the model."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ultralytics import YOLO
from ultralytics.nn import tasks

from models.modules import CAFM, DMMM, GradientAwareDecoupledDetectionHead


def register_hagnet_modules() -> None:
    """Expose the custom layers to the Ultralytics YAML model parser."""
    tasks.DMMM = DMMM
    tasks.CAFM = CAFM
    tasks.GradientAwareDecoupledDetectionHead = GradientAwareDecoupledDetectionHead
    if GradientAwareDecoupledDetectionHead not in tasks.DETECT_CLASS:
        tasks.DETECT_CLASS = (*tasks.DETECT_CLASS, GradientAwareDecoupledDetectionHead)


def build_hagnet(config: str | Path | None = None) -> YOLO:
    """Build HAG-Net from its public YAML configuration."""
    register_hagnet_modules()
    config = Path(config) if config else Path(__file__).with_name("HAG-Net.yaml")
    return YOLO(str(config))


if __name__ == "__main__":
    model = build_hagnet()
    print("HAG-Net model build passed.")
