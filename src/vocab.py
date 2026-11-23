import json
from pathlib import Path
from llm_sdk import Small_LLM_Model


def load_vocab(model: Small_LLM_Model) -> dict[int, str]:
    vocab_path = Path(model.get_path_to_vocab_file())
    with vocab_path.open("r") as file:
        vocab = json.load(file)
    return {
        int(token_id): token
        for token, token_id in vocab.items()
    }
