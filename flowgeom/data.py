from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class ImageRecord:
    image_path: str
    label: int
    generator_name: str
    split: str


def _validate_df(df: pd.DataFrame) -> pd.DataFrame:
    required = ["image_path", "label", "generator_name", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset manifest missing columns: {missing}")
    return df[required].copy()


def build_manifest_from_folder(root: str) -> pd.DataFrame:
    root = Path(root)
    rows: list[ImageRecord] = []
    real_dir = root / "real"
    if real_dir.exists():
        for p in real_dir.rglob("*"):
            if p.suffix.lower() in IMG_EXTS:
                rows.append(ImageRecord(str(p), 0, "real", "train"))
    fake_root = root / "fake"
    if fake_root.exists():
        for gen_dir in fake_root.iterdir():
            if gen_dir.is_dir():
                for p in gen_dir.rglob("*"):
                    if p.suffix.lower() in IMG_EXTS:
                        rows.append(ImageRecord(str(p), 1, gen_dir.name, "train"))
    return pd.DataFrame([r.__dict__ for r in rows])


class FlowGeomDataset(Dataset):
    """Dataset returning tensorized images and metadata for surrogate probing."""

    def __init__(self, df: pd.DataFrame, image_size: int = 256, image_root: Optional[str] = None):
        self.df = _validate_df(df).reset_index(drop=True)
        self.image_root = Path(image_root) if image_root else None
        self.tf = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = Path(row.image_path)
        if self.image_root and not path.is_absolute():
            path = self.image_root / path
        if path.exists():
            x = self.tf(Image.open(path).convert("RGB"))
        else:
            x = torch.rand(3, 64, 64)
        return x, {
            "image_path": str(path),
            "label": int(row.label),
            "generator_name": str(row.generator_name),
            "split": str(row.split),
        }


def load_dataset(data_csv: Optional[str], data_root: Optional[str], image_size: int = 256) -> FlowGeomDataset:
    """Load folder/CSV dataset, or create tiny synthetic fallback for debug pipeline runs."""
    if data_csv:
        df = _validate_df(pd.read_csv(data_csv))
    elif data_root:
        df = _validate_df(build_manifest_from_folder(data_root))
    else:
        df = pd.DataFrame(
            [{"image_path": f"synthetic_{i}.png", "label": i % 2, "generator_name": "real" if i % 2 == 0 else "dummy_fake", "split": "train" if i < 6 else ("val" if i < 8 else "test")} for i in range(10)]
        )
    return FlowGeomDataset(df, image_size=image_size, image_root=data_root)
