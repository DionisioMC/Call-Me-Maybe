import json
import re
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
        prompt: str,
        function: FunctionDef
    ) -> dict[str, object]:
        params: dict[str, Any] = {}
        parameter_items = list(function.parameters.items())
        for index, (name, definition) in enumerate(parameter_items):
            value: Any
            is_last = index == len(parameter_items) - 1
            allowed_terminators = {"}"} if is_last else {","}
            if definition.type in ("number", "integer", "float"):
                value = self.generate_number(llm_service, prompt,
                                             definition.type,
                                             allowed_terminators)
            elif definition.type == "boolean":
                value = self.generate_boolean(llm_service, prompt)
            elif definition.type == "string":
                value = self.generate_string(llm_service, prompt)
            else:
                raise ValueError(
                    f"Unsupported parameter type: {definition.type}")
            params[name] = value
        return params

    @staticmethod
    def is_valid_number_prefix(
        text: str,
        number_type: str
    ) -> bool:
        text = text.strip()
        if number_type == "integer":
            pattern = r"-?(?:0|[1-9][0-9]*)?"
        else:
            pattern = (
                r"-?(?:(?:0|[1-9][0-9]*)"
                r"(?:\.[0-9]*)?"
                r"(?:[eE][+-]?[0-9]*)?)?"
            )
        return re.fullmatch(pattern, text) is not None

    @staticmethod
    def is_complete_number(
        text: str,
        number_type: str
    ) -> bool:
        text = text.strip()
        if number_type == "integer":
            pattern = r"-?(?:0|[1-9][0-9]*)"
        else:
            pattern = (
                r"-?(?:0|[1-9][0-9]*)"
                r"(?:\.[0-9]+)?"
                r"(?:[eE][+-]?[0-9]+)?"
            )
        return re.fullmatch(pattern, text) is not None

    def generate_number(
        self,
        llm_service: LLMService,
        prompt: str,
        number_type: str,
        allowed_terminators: set[str]
    ) -> int | float:
        if number_type not in ("number", "integer", "float"):
            raise ValueError(f"Unsupported number type: {number_type}")

        if not allowed_terminators or not allowed_terminators <= {",", "}"}:
            raise ValueError("Invalid number terminators.")
        (
            valid_tokens,
            valid_tokens_last,
            minus_tokens,
            point_tokens
        ) = self.token_manager.valid_tokens_num
        candidate_ids = (
            set(valid_tokens)
            | set(valid_tokens_last)
            | set(minus_tokens)
        )
        generated_ids: list[int] = []
        generated_text = ""
        for _ in range(32):
            continuation_ids = []
            terminating_ids = {}
            for token_id in candidate_ids:
                token_text = llm_service.decode([token_id])
                terminator = next(
                    (
                        ending for ending in allowed_terminators
                        if token_text.endswith(ending)
                    ),
                    None
                )
                number_piece = (
                    token_text[:-1]
                    if terminator is not None
                    else token_text
                )
                candidate = generated_text + number_piece
                if terminator is not None:
                    if self.is_complete_number(candidate, number_type):
                        terminating_ids[token_id] = terminator
                elif self.is_valid_number_prefix(candidate, number_type):
                    if (
                        number_type == "integer"
                        and token_id in point_tokens
                    ):
                        continue
                    continuation_ids.append(token_id)
            allowed_ids = set(continuation_ids) | set(terminating_ids)
            if not allowed_ids:
                raise RuntimeError(
                    f"No valid numerical tokens for prefix "
                    f"{generated_text!r}"
                )
            input_ids = (
                llm_service.encode(prompt) + generated_ids
            )
            logits = llm_service.get_logits(input_ids)
            allowed_ids = {
                token_id for token_id in allowed_ids
                if token_id < len(logits) and isfinite(logits[token_id])
            }
            if not allowed_ids:
                raise RuntimeError("No valid logits available.")
            next_token = max(
                allowed_ids,
                key=lambda token_id: logits[token_id]
            )
            token_text = llm_service.decode([next_token])
            if next_token in terminating_ids:
                try:
                    if number_type == "integer":
                        return int(generated_text)
                    result = float(generated_text)
                    if not isfinite(result):
                        raise ValueError("Generated number is not finite.")
                    return result
                except ValueError as error:
                    raise ValueError(
                        f"Invalid generated number: {generated_text!r}"
                    ) from error
            generated_ids.append(next_token)
            generated_text += token_text
        raise RuntimeError(
            f"Maximum generation length reached: {generated_text!r}"
        )

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
        max_tokens: int = 64
    ) -> str:
        valid_tokens = self.token_manager.valid_tokens_str
        generated_ids: list[int] = []
        for _ in range(max_tokens):
            input_ids = llm_service.encode(prompt)
            input_ids.extend(generated_ids)
            logits = llm_service.get_logits(input_ids)
            valid_ids = [
                token_id for token_id in valid_tokens
                if token_id < len(logits)
            ]
            if not valid_ids:
                raise RuntimeError(
                    "No valid tokens available for string generation."
                )
            next_token = max(
                valid_ids,
                key=lambda token_id: logits[token_id]
            )
            generated_ids.append(next_token)
            generated_text = llm_service.decode(generated_ids)
            candidate = '"' + generated_text
            try:
                value = json.loads(candidate)
                if isinstance(value, str):
                    return value
            except json.JSONDecodeError:
                continue
        raise RuntimeError("Maximum string generation length reached. ")
