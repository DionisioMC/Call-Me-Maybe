from pydantic import BaseModel, PrivateAttr

from llm_sdk import Small_LLM_Model


class LLMService(BaseModel):
    _llm: Small_LLM_Model = PrivateAttr(default_factory=Small_LLM_Model)

    def encode(self, text: str) -> list[int]:
        return self._llm.encode(text).squeeze(0).tolist()

    def decode(self, token_ids: list[int]) -> str:
        return self._llm.decode(token_ids)

    def get_logits(self, input_ids: list[int]) -> list[float]:
        return self._llm.get_logits_from_input_ids(input_ids)

    def get_vocab_path(self) -> str:
        return self._llm.get_path_to_vocab_file()

    def find_max_logit_id(self, prompt: str,
                          allowed_ids: set[int]) -> int:
        logits: list[float] = (self.get_logits(
                self._llm.encode(prompt).tolist()[0]))
        return max(allowed_ids, key=lambda token_id: logits[token_id])
