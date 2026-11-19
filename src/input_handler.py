import json
from pathlib import Path

from pydantic import ValidationError

from src.models import Prompt


def load_prompts(file_path: str) -> list[Prompt]:
    path = Path(file_path)
    try:
        with path.open("r") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {path}")
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}")
    if not isinstance(data, list):
        raise ValueError("The input file must contain a JSON array.")
    try:
        prompts = [Prompt.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"Invalid prompt structure: {error}")
    return prompts
