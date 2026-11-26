from pydantic import BaseModel
from .models import FunctionDef
from .llm_service import LLMService


class FunctionSelector(BaseModel):
    functions: list[FunctionDef]

    def get_function_tokens(
        self,
        llm_service: LLMService
    ) -> dict[str, list[int]]:
        return {
            function.name: llm_service.encode(function.name)
            for function in self.functions}

    def filter_functions(self, generated_tokens: list[int],
                         function_tokens: dict[str, list[int]]
                         ) -> dict[str, list[int]]:
        return {
            name: tokens for name, tokens in function_tokens.items()
            if tokens[:len(generated_tokens)] == generated_tokens}

    def select_function(self, llm_service: LLMService, prompt: str
                        ) -> FunctionDef:
        if not self.functions:
            raise ValueError("No functions available.")
        function_tokens = self.get_function_tokens(llm_service)
        candidates = dict(function_tokens)
        generated_tokens: list[int] = []
        quote_tokens: list[int] = llm_service.encode('"')
        if len(quote_tokens) != 1:
            raise RuntimeError(
                "The quotation mark must be represented by one token.")
        quote_id: int = quote_tokens[0]
        max_length = max(len(tokens) for tokens in candidates.values())
        prompt_tokens = llm_service.encode(prompt)
        for position in range(max_length + 1):
            allowed_ids: set[int] = set()
            can_end = False
            for tokens in candidates.values():
                if len(tokens) == position:
                    can_end = True
                else:
                    allowed_ids.add(tokens[position])
            if can_end:
                allowed_ids.add(quote_id)
            if not allowed_ids:
                raise RuntimeError(
                    "No valid tokens available for function selection.")
            logits = llm_service.get_logits(prompt_tokens + generated_tokens)
            next_token = max(allowed_ids,
                             key=lambda token_id: logits[token_id])
            if next_token == quote_id and can_end:
                selected_tokens = generated_tokens
                selected_name = next(
                    (name for name, tokens in function_tokens.items()
                        if tokens == selected_tokens),
                    None
                )
                if selected_name is None:
                    raise RuntimeError(
                        "Could not identify the selected function.")
                return next(
                    function for function in self.functions
                    if function.name == selected_name)
            generated_tokens.append(next_token)
            candidates = self.filter_functions(generated_tokens, candidates)
            if not candidates:
                raise RuntimeError(
                    "No functions match the generated tokens.")
        raise RuntimeError(
            "Function selection exceeded the maximum token length.")
