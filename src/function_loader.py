import json
from pathlib import Path
from pydantic import ValidationError
from src.models import FunctionDef


def load_functions(file_path: str) -> list[FunctionDef]:
    path = Path(file_path)
    try:
        with path.open("r") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Function definitions file not found: {path}"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path}: {error}"
        ) from error
    if not isinstance(data, list):
        raise ValueError(
            "Function definitions must be a JSON array."
        )
    try:
        functions = [
            FunctionDef.model_validate(item)
            for item in data
        ]
    except ValidationError as error:
        raise ValueError(
            f"Invalid function definition: {error}"
        ) from error
    function_names = [function.name for function in functions]
    if len(function_names) != len(set(function_names)):
        raise ValueError(
            "Duplicate function names found."
        )
    return functions
