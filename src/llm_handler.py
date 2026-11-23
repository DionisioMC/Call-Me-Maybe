from llm_sdk import Small_LLM_Model


def load_model() -> Small_LLM_Model:
    model = Small_LLM_Model()
    return model


def encode_prompt(model: Small_LLM_Model, prompt: str) -> list[int]:
    encoded = model.encode(prompt)
    input_ids = encoded.squeeze(0).tolist()
    return [int(token_id) for token_id in input_ids]


def get_logits(model: Small_LLM_Model, input_ids: list[int]) -> list[float]:
    if not input_ids:
        raise ValueError("Cannot retrieve logits from empty input.")
    logits = model.get_logits_from_input_ids(input_ids)
    if not logits:
        raise RuntimeError("The model returned empty logits.")
    return logits
