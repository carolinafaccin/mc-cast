from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

URBAN = 3  # model class id of urban area (see config.yaml)
WATER = 4
NODATA = 0


@dataclass
class Config:
    raw: dict

    def __getitem__(self, key):
        return self.raw[key]

    @property
    def classes(self) -> dict[int, dict]:
        return {int(k): v for k, v in self.raw["classes"].items()}

    @property
    def n_classes(self) -> int:
        return len(self.raw["classes"])


def load(path: str | Path | None = None) -> Config:
    path = Path(path) if path else ROOT / "config.yaml"
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
