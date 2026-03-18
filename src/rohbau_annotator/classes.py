"""Semantic class definitions for Rohbau3D construction site annotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SemanticClass:
    """A single semantic class for construction site elements."""

    id: int
    name: str
    color: Tuple[int, int, int]  # RGB 0-255


# 13 semantic classes for construction site annotation.
# ID 0 is reserved for unlabeled / background.
CLASSES: List[SemanticClass] = [
    SemanticClass(0, "unlabeled", (0, 0, 0)),
    SemanticClass(1, "wall", (174, 199, 232)),
    SemanticClass(2, "floor", (152, 223, 138)),
    SemanticClass(3, "ceiling", (31, 119, 180)),
    SemanticClass(4, "column", (255, 187, 120)),
    SemanticClass(5, "beam", (188, 189, 34)),
    SemanticClass(6, "door", (140, 86, 75)),
    SemanticClass(7, "window", (255, 152, 150)),
    SemanticClass(8, "pipe", (214, 39, 40)),
    SemanticClass(9, "duct", (197, 176, 213)),
    SemanticClass(10, "cable_tray", (148, 103, 189)),
    SemanticClass(11, "rebar", (196, 156, 148)),
    SemanticClass(12, "formwork", (23, 190, 207)),
    SemanticClass(13, "other", (127, 127, 127)),
]

CLASS_BY_ID: Dict[int, SemanticClass] = {c.id: c for c in CLASSES}
CLASS_BY_NAME: Dict[str, SemanticClass] = {c.name: c for c in CLASSES}

# Number of annotatable classes (excluding unlabeled)
NUM_CLASSES = len(CLASSES) - 1


def class_color_norm(cls: SemanticClass) -> Tuple[float, float, float]:
    """Return class color normalized to [0, 1] for matplotlib."""
    return (cls.color[0] / 255.0, cls.color[1] / 255.0, cls.color[2] / 255.0)
