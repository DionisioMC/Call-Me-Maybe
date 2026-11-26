import json
from pydantic import BaseModel
from .models import FunctionDef
from .llm_service import LLMService
from .token_manager import TokenManager
from typing import Any
from math import isfinite


class ParameterGenerator(BaseModel):
    token_manager: TokenManager

    def generate_parameters(
        self,
        llm_service: LLMService,
        function: FunctionDef,
        prompt: str
    ) -> dict[str, object]:
        parameters: dict[str, Any] = {}
        parameter_items = list(function.parameters.items())
        for index, (name, definition) in enumerate(parameter_items):
            value: Any
            is_last: bool = index == (len(parameter_items) - 1)
            if definition.type in ("number", "integer", "float"):
                value = self.generate_number(llm_service, prompt,
                                             definition.type, is_last)
            elif definition.type == "boolean":
                value = self.generate_boolean(llm_service, prompt)
            elif definition.type == "string":
                value = self.generate_string(llm_service, prompt, is_last)
            else:
                raise ValueError(
                    f"Unsupported parameter type: {definition.type}")
            parameters[name] = value
        return parameters

    def generate_number(
        self,
        llm_service: LLMService,
        prompt: str,
        number_type: str,
        is_last_param: bool,
        max_tokens: int = 32
    ) -> int | float:
        if number_type not in ("number", "integer", "float"):
            raise ValueError(f"Unsupported number type: {number_type}")
        (valid_tokens, valid_tokens_last, valid_minus_tokens,
            point_tokens) = self.token_manager.valid_tokens_num
        allowed_tokens = set(valid_tokens_last
                             if is_last_param else valid_tokens)
        generated_text = ""
        for _ in range(max_tokens):
            current_tokens = allowed_tokens.copy()
            if not generated_text.strip():
                current_tokens.update(valid_minus_tokens)
            if number_type == "integer" or "." in generated_text:
                current_tokens.difference_update(point_tokens)
            if not current_tokens:
                raise RuntimeError("No valid numerical tokens available.")
            input_ids = llm_service.encode(prompt + generated_text)
            logits = llm_service.get_logits(input_ids)
            valid_ids = [
                token_id for token_id in current_tokens
                if token_id < len(logits) and isfinite(logits[token_id])
            ]
            if not valid_ids:
                raise RuntimeError("No valid logits available.")
            next_token = max(valid_ids, key=lambda token_id: logits[token_id])
            token_text = llm_service.decode([next_token])
            generated_text += token_text
            terminator = "}" if is_last_param else ","
            if generated_text.rstrip().endswith(terminator):
                number_text = generated_text.rstrip()[:-1].strip()
                try:
                    if number_type == "integer":
                        return int(number_text)
                    return float(number_text)
                except ValueError:
                    raise ValueError(
                        f"Generated invalid number: {number_text}")
        raise RuntimeError("Maximum number of generation tokens reached.")

    def generate_boolean(self, llm_service: LLMService, prompt: str) -> bool:
        true_tokens, false_tokens = (self.token_manager.valid_tokens_bool)
        if not true_tokens or not false_tokens:
            raise RuntimeError("Boolean tokens are unavailable.")
        if true_tokens[0] == false_tokens[0]:
            raise RuntimeError(
                "True and false share their first token. "
                "A multi-token selection algorithm is required.")
        logits = llm_service.get_logits(llm_service.encode(prompt))
        candidates = {
            true_tokens[0]: True,
            false_tokens[0]: False
        }
        valid_ids = [token_id for token_id in candidates
                     if token_id < len(logits)]
        if not valid_ids:
            raise RuntimeError("No valid boolean tokens available.")
        selected_token = max(valid_ids, key=lambda token_id: logits[token_id])
        return candidates[selected_token]

    def generate_string(
        self,
        llm_service: LLMService,
        prompt: str,
        is_last_parameter: bool,
        max_tokens: int = 64
    ) -> str:
        if is_last_parameter:
            valid_tokens = (self.token_manager.valid_tokens_str[1])
        else:
            valid_tokens = (self.token_manager.valid_tokens_str[0])
        generated_text = '"'
        for _ in range(max_tokens):
            logits = llm_service.get_logits(llm_service.encode(
                prompt + generated_text))
            valid_ids = [token_id for token_id in valid_tokens
                         if token_id < len(logits)]
            if not valid_ids:
                raise RuntimeError("No valid string tokens available.")
            next_token = max(valid_ids, key=lambda token_id: logits[token_id])
            token_text = llm_service.decode([next_token])
            generated_text += token_text
            candidates = [generated_text]
            if generated_text.endswith('",') or generated_text.endswith('"}'):
                candidates.insert(0, generated_text[:-1])
            for candidate in candidates:
                try:
                    value = json.loads(candidate)
                    if isinstance(value, str):
                        return value
                except json.JSONDecodeError:
                    continue
        raise RuntimeError("Maximum string generation length reached.")
