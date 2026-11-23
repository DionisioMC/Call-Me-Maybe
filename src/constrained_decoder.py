from math import inf


def mask_logits(
    logits: list[float],
    token_pieces: dict[int, str],
    grammar,
    state
) -> tuple[list[float], dict]:
    masked_logits = [-inf] * len(logits)
    candidate_states = {}
    for token_id, piece in token_pieces.items():
        if token_id >= len(logits):
            continue
        next_state = grammar.advance(state, piece)
        if next_state is None:
            continue
        masked_logits[token_id] = logits[token_id]
        candidate_states[token_id] = next_state
    return masked_logits, candidate_states
