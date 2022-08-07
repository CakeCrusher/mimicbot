from typing import TypedDict
from pathlib import Path
from enum import Enum


class ModelSave(TypedDict):
    url: str
    context_length: int
    data_path: Path  # its actually session_path


class Platform(Enum):
    DISCORD = "discord"
    NONE = "none"
