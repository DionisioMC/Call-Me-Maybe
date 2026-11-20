import json
from typing import TypeVar
from .models import FunctionDef, Prompt
from pydantic import BaseModel, ValidationError


TV = TypeVar("TV", bound=BaseModel)


def load_json(file_path: str, model_class: type[TV]) -> list[TV]:
    try:
        with open(file_path, "r") as file:
            content = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    if not isinstance(content, list):
        raise ValueError(f"{file_path} must contain a JSON array")
    if not content:
        raise ValueError(f"{file_path} is empty")
    errors = ""
    list_model_class = []
    for item in content:
        try:
            item_validate = model_class.model_validate(item)
            list_model_class.append(item_validate)
        except ValidationError as e:
            for error in e.errors():
                errors += f"Validation error in {file_path}: {error['msg']}\n"
    if errors != "":
        raise Exception(errors)
    return list_model_class


def parse(argv: list[str]) -> tuple[list[FunctionDef], list[Prompt], str]:
    if (len(argv) == 7 and (argv[1] != "--functions_definition" or
                            argv[3] != "--input" or
                            argv[5] != "--output")):
        raise Exception("Usage: uv run python -m src "
                        "[--functions_definition <function_definition_file>] "
                        "[--input <input_file>] [--output <output_file>] "
                        "arguments in square brackets are optional")
    if len(argv) == 1:
        functions_definition_path = "data/input/functions_definition.json"
        input_path = "data/input/function_calling_tests.json"
        output_path = "data/output/function_calls.json"
    else:
        if argv[1] == "--functions_definition":
            functions_definition_path = argv[2]
        elif argv[3] == "--functions_definition":
            functions_definition_path = argv[4]
        elif argv[5] == "--functions_definition":
            functions_definition_path = argv[6]
        else:
            functions_definition_path = "data/input/functions_definition.json"
        if argv[1] == "--input":
            input_path = argv[2]
        elif argv[3] == "--input":
            input_path = argv[4]
        elif argv[5] == "--input":
            input_path = argv[6]
        else:
            input_path = "data/input/function_calling_tests.json"
        if argv[1] == "--output":
            output_path = argv[2]
        elif argv[3] == "--output":
            output_path = argv[4]
        elif argv[5] == "--output":
            output_path = argv[6]
        else:
            output_path = "data/output/function_calls.json"
    functions_json_data: list[FunctionDef] = load_json(
        functions_definition_path, FunctionDef)
    tests_json_data: list[Prompt] = load_json(input_path, Prompt)
    return (functions_json_data, tests_json_data, output_path)
