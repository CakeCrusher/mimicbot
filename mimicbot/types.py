from typing import TypedDict
from pathlib import Path


class ModelSave(TypedDict):
    url: str
    context_length: int
    data_path: Path # its actually session_path
