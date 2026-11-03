"""Offline loader for the fixed MNIST subset used in variant 4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARRAY_KEYS = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")

def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def load_dataset(name: str = "mnist", data_dir: str | Path | None = None, verify: bool = True) -> dict:
    name = name.lower().replace("-", "").replace("_", "")
    if name != "mnist":
        raise ValueError("Variant 4 uses only MNIST")
    directory = PROJECT_ROOT / "data" if data_dir is None else Path(data_dir)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["datasets"]["mnist"]
    path = directory / metadata["file"]
    if verify and file_sha256(path) != metadata["sha256"]:
        raise ValueError(f"Checksum mismatch for {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in ARRAY_KEYS}
    return {**arrays, "metadata": metadata, "class_names": metadata["class_names"],
            "image_shape": metadata["image_shape"], "pixel_max": metadata["pixel_max"]}
