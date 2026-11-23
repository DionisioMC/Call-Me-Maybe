import json
from typing import Any
from .models import FunctionDef, PromptInput


def gen_prompt(
    prompt: PromptInput,
    functions: list[FunctionDef]
) -> str:
    function_descriptions = []
    for function in functions:
        description: dict[str, Any] = {
            "name": function.name,
            "description": function.description,
            "parameters": {
                name: {"type": parameter.type}
                for name, parameter in function.parameters.items()
            },
            "returns": {
                "type": function.returns.type
            }
        }
        function_descriptions.append(description)
    functions_json: str = json.dumps(function_descriptions, indent=4)
    return (f"""
You are a function calling assistant.
Your task is to identify the function that best matches the user's request.
Available functions:
{functions_json}
User request:
{prompt.prompt}
Identify the appropriate function and extract all its required arguments.
Do not execute the function.
Do not answer the user's request directly.
Return the function name and its arguments.""")
