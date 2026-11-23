from .parser import parse
from .prompt_generator import gen_prompt
from .vocab import load_vocab
from .llm_handler import load_model, encode_prompt, get_logits
from sys import argv
from .llm_handler import Small_LLM_Model


def main() -> None:
    try:
        functions, prompts, output_file = parse(argv)
        model: Small_LLM_Model = load_model()
        vocabulary = load_vocab(model)
        print(f"Vocabulary size: {len(vocabulary)}")
        for token_id, token in list(vocabulary.items())[:20]:
            print(token_id, repr(token))
    except ValueError as e:
        print(e)
    except Exception as e:
        print(e)


if __name__ == "__main__":
    main()
