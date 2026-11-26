import json
from math import isfinite
from pathlib import Path
from pydantic import BaseModel, PrivateAttr
from .llm_service import LLMService


class TokenManager(BaseModel):
    id_content_vocab: dict[int, str]
    valid_tokens_num: tuple[
        list[int],
        list[int],
        list[int],
        list[int]]
    valid_tokens_bool: tuple[list[int], list[int]]
    valid_tokens_str: tuple[list[int], list[int]]
    _id_content_vocab: dict[int, str] = PrivateAttr()

    @classmethod
    def from_model(cls, llm: LLMService) -> "TokenManager":
        vocab_path = Path(llm.get_vocab_path())
        try:
            with vocab_path.open("r") as file:
                vocabulary: dict[str, int] = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary not found: {vocab_path}")
        cls._id_content_vocab = {
            token_id: token
            for token, token_id in vocabulary.items()}
        valid_tokens_num = cls.get_valid_tokens_num(
            cls._id_content_vocab)
        valid_tokens_bool = cls.get_valid_tokens_bool(llm)
        valid_tokens_str = cls.get_valid_tokens_str(
            cls._id_content_vocab)
        return cls(
            id_content_vocab=cls._id_content_vocab,
            valid_tokens_num=valid_tokens_num,
            valid_tokens_bool=valid_tokens_bool,
            valid_tokens_str=valid_tokens_str)

    @staticmethod
    def get_valid_tokens_num(
        vocabulary: dict[int, str]
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        valid_minus_token: list[int] = []
        valid_tokens: list[int] = []
        valid_tokens_last: list[int] = []
        point_tokens: list[int] = []
        for id, token in vocabulary.items():
            flag_minus = False
            flag_middle = False
            flag_last = False
            try:
                check_token = float(token)
                if not isfinite(check_token):
                    raise
                if token.count(".") == 1:
                    point_tokens.append(id)
                if check_token < 0:
                    valid_minus_token.append(id)
                else:
                    valid_tokens.append(id)
                    valid_tokens_last.append(id)
            except Exception:
                if "-" in token and token.count("-") == 1 and token[0] == "-":
                    if token == "-":
                        valid_minus_token.append(id)
                        continue
                    token = token[1:]
                    flag_minus = True
                if "," in token and token.count(",") == 1 and token[-1] == ",":
                    if token == "," and not flag_minus:
                        valid_tokens.append(id)
                        continue
                    token = token[:-1]
                    flag_middle = True
                elif ("}" in token and token.count("}") == 1 and
                        token[-1] == "}"):
                    if token == "}" and not flag_minus:
                        valid_tokens_last.append(id)
                        continue
                    token = token[:-1]
                    flag_last = True
                if token.count(".") == 1:
                    if (token == "." and not flag_minus and
                            not flag_middle and not flag_last):
                        valid_tokens.append(id)
                        valid_tokens_last.append(id)
                        point_tokens.append(id)
                        continue
                    token_split = token.split(".")
                    try:
                        check_token = float(token_split[0])
                        if not isfinite(check_token):
                            raise
                        check_token = float(token_split[1])
                        if not isfinite(check_token):
                            raise
                        if flag_minus:
                            valid_minus_token.append(id)
                        valid_tokens.append(id)
                        valid_tokens_last.append(id)
                        point_tokens.append(id)
                        continue
                    except Exception:
                        pass
                try:
                    check_token = float(token)
                    if not isfinite(check_token):
                        raise
                    if flag_minus:
                        valid_minus_token.append(id)
                    if flag_middle:
                        valid_tokens.append(id)
                    if flag_last:
                        valid_tokens_last.append(id)
                except Exception:
                    pass
        return (valid_tokens, valid_tokens_last,
                valid_minus_token, point_tokens)

    @staticmethod
    def get_valid_tokens_bool(
        llm: LLMService
    ) -> tuple[list[int], list[int]]:
        true_tokens = llm.encode("true")
        false_tokens = llm.encode("false")
        return true_tokens, false_tokens

    @staticmethod
    def get_valid_tokens_str(
        vocabulary: dict[int, str]
    ) -> tuple[list[int], list[int]]:
        valid_tokens = []
        valid_tokens_last = []
        for id, token in vocabulary.items():
            if (token.count('"') == 1 and
                (token[-1] == '"' or
                    (token[-2] == '"' and token[-1] == ","))):
                valid_tokens.append(id)
            elif (token.count('"') == 1 and
                    token[-2] == '"' and token[-1] == "}"):
                valid_tokens_last.append(id)
            elif token.count('"') == 0:
                valid_tokens.append(id)
                valid_tokens_last.append(id)
        return (valid_tokens, valid_tokens_last)
