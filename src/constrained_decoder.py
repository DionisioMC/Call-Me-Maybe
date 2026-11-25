from math import isfinite
import json
from time import time
import os
from llm_sdk import Small_LLM_Model
from .models import FunctionDef, PromptInput
from .prompt_generator import gen_prompt
from pydantic import BaseModel, PrivateAttr


class ConstrainedDecoder(BaseModel):
    functions_json_data: list[FunctionDef]
    tests_json_data: list[PromptInput]
    output_path: str
    terminal_output: str = ""
    _llm: Small_LLM_Model = PrivateAttr(default_factory=Small_LLM_Model)
    _id_content_vocab: dict[int, str] = PrivateAttr()
    _function_names: list[str] = PrivateAttr(default=[])
    _valid_tokens_num: tuple[
        list[int], list[int], list[int],
        list[int]] = PrivateAttr()
    _valid_tokens_bool: tuple[
        list[int], list[int]] = PrivateAttr()
    _valid_tokens_str: tuple[
        list[int], list[int]] = PrivateAttr()

    def model_post_init(self, context) -> None:
        self.create_vocab_file()
        self.get_function_names()
        self._valid_tokens_num = self.get_valid_tokens_num()
        self._valid_tokens_bool = self.get_valid_tokens_bool()
        self._valid_tokens_str = self.get_valid_tokens_str()

    def create_vocab_file(self) -> None:
        vocab_file_path: str = self._llm.get_path_to_vocab_file()
        try:
            with open(vocab_file_path, "r") as file:
                vocab_file: dict = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocab file not found: {vocab_file_path}")
        self._id_content_vocab = {value: key for key, value
                                  in vocab_file.items()}

    def get_function_names(self) -> None:
        for function in self.functions_json_data:
            self._function_names.append(function.name)

    def find_max_logit_id(self, prompt: str,
                          allowed_ids: set[int]) -> int:
        logits: list[float] = (self._llm.get_logits_from_input_ids(
                self._llm.encode(prompt).tolist()[0]))
        return max(allowed_ids, key=lambda token_id: logits[token_id])

    def get_valid_tokens_num(self) -> tuple[list[int],
                                            list[int], list[int],
                                            list[int]]:
        valid_minus_token: list[int] = []
        valid_tokens: list[int] = []
        valid_tokens_last: list[int] = []
        point_tokens: list[int] = []
        for id, token in self._id_content_vocab.items():
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

    def get_valid_tokens_bool(self) -> tuple[list[int], list[int]]:
        valid_tokens_true = self._llm.encode("true").tolist()[0]
        valid_tokens_false = self._llm.encode("false").tolist()[0]
        return (valid_tokens_true, valid_tokens_false)

    def get_valid_tokens_str(self) -> tuple[list[int], list[int]]:
        valid_tokens = []
        valid_tokens_last = []
        for id, token in self._id_content_vocab.items():
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

    def del_invalid_function(self, tokens_writed: list[int],
                             function_ids: dict[str, list[int]]
                             ) -> dict[str, list[int]]:
        function_ids_copy = dict(function_ids)
        for i in range(len(tokens_writed)):
            for key, value in function_ids.items():
                if i >= len(value) or value[i] != tokens_writed[i]:
                    function_ids_copy.pop(key, None)
        return dict(function_ids_copy)

    def del_functions_json_data(self, function_ids: dict[str, list[int]],
                                functions_json_data_prompt: list[
                                FunctionDef]) -> None:
        functions_json_data_copy = list(functions_json_data_prompt)
        for function in functions_json_data_copy:
            if function.name not in function_ids:
                functions_json_data_prompt.remove(function)

    def valid_tokens_functions(self) -> dict[str, list[int]]:
        functions_ids: dict[str, list[int]] = {}
        for function in self._function_names:
            functions_ids[function] = (self._llm.encode(
                function).tolist()[0])
        return functions_ids

    def write_function(self, partial_prompt: str, prompt_output: str,
                       function_ids: dict[str, list[int]],
                       functions_json_data_prompt: list[FunctionDef],
                       user_prompt: str) -> tuple[str, str]:
        new_function_ids = dict(function_ids)
        biggest_function = max(len(ids) for ids in new_function_ids.values())
        wrote_function_ids = []
        id_quote = self._llm.encode('"').tolist()[0][0]
        id_quote_comma = self._llm.encode('",').tolist()[0][0]
        for i in range(biggest_function):
            allowed_ids = set()
            for id in new_function_ids.values():
                if len(id) == i:
                    allowed_ids.add(id_quote)
                    allowed_ids.add(id_quote_comma)
                else:
                    allowed_ids.add(id[i])
            if len(new_function_ids) > 1:
                biggest_logit_id = self.find_max_logit_id(
                    partial_prompt + prompt_output, allowed_ids)
            else:
                for bli in allowed_ids:
                    biggest_logit_id = bli
            choosed_token = self._llm.decode([biggest_logit_id])
            prompt_output += choosed_token
            print("\n" * 50 + self.terminal_output + prompt_output)
            wrote_function_ids.append(biggest_logit_id)
            if prompt_output[-2:] == '",' or prompt_output[-1] == '"':
                if prompt_output[-2:] == '",':
                    prompt_output = prompt_output[:-2]
                else:
                    prompt_output = prompt_output[:-1]
                function_name = prompt_output.split('"')[-1].strip()
                functions_json_data_copy = list(functions_json_data_prompt)
                for function in functions_json_data_copy:
                    if function_name != function.name:
                        functions_json_data_prompt.remove(function)
                partial_prompt = gen_prompt(
                    functions_json_data_prompt, user_prompt)
                break
            if len(new_function_ids) > 1:
                new_function_ids = self.del_invalid_function(
                    wrote_function_ids, new_function_ids)
                self.del_functions_json_data(
                    new_function_ids, functions_json_data_prompt)
                partial_prompt = gen_prompt(
                    functions_json_data_prompt, user_prompt)
                biggest_function = max(len(ids)
                                       for ids in new_function_ids.values())
        return (partial_prompt, prompt_output)

    def write_parameters(self, partial_prompt: str, prompt_output: str,
                         functions_json_data_prompt: list[FunctionDef]) -> str:
        parameters = functions_json_data_prompt[0].parameters
        (valid_tokens_numbers, valid_tokens_numbers_last,
         valid_minus_token, point_tokens) = self._valid_tokens_num
        (valid_tokens_string,
         valid_tokens_string_last) = self._valid_tokens_str
        (valid_tokens_bool_true,
         valid_tokens_bool_false) = self._valid_tokens_bool
        for parameter, value in parameters.items():
            if parameter != next(iter(parameters)):
                prompt_output += " "
            prompt_output += f"\"{parameter}\": "
            if (value.type == "number" or value.type == "integer"
                    or value.type == "float"):
                if parameter != next(reversed(parameters)):
                    if parameter == next(iter(parameters)):
                        valid_tokens = (valid_tokens_numbers.copy() +
                                        valid_minus_token.copy())
                    else:
                        valid_tokens = valid_tokens_numbers.copy()
                    removed_points = False
                    parameter_output = ""
                    for i in range(32):
                        if (not removed_points and
                            (value.type == "integer" or
                             prompt_output.split(":")[-1].count(".") >= 1)):
                            for token in point_tokens:
                                if token in valid_tokens:
                                    valid_tokens.remove(token)
                            removed_points = True
                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output, set(valid_tokens))
                        choosed_token = self._llm.decode([biggest_logit_id])
                        prompt_output += choosed_token
                        parameter_output += choosed_token
                        print("\n" * 50 + self.terminal_output + prompt_output)
                        if prompt_output[-1] == ",":
                            if (value.type != "integer" and
                                    "." not in parameter_output):
                                prompt_output = prompt_output[:-1]
                                prompt_output += ".0,"
                            break
                    if i == 31 and prompt_output[-1] != ",":
                        prompt_output += ","
                else:
                    if parameter == next(iter(parameters)):
                        valid_tokens = (valid_tokens_numbers_last.copy() +
                                        valid_minus_token.copy())
                    else:
                        valid_tokens = valid_tokens_numbers_last.copy()
                    removed_points = False
                    parameter_output = ""
                    for i in range(32):
                        if (not removed_points and
                            (value.type == "integer" or
                             prompt_output.split(":")[-1].count(".") >= 1)):
                            for token in point_tokens:
                                if token in valid_tokens:
                                    valid_tokens.remove(token)
                            removed_points = True
                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output, set(valid_tokens))
                        choosed_token = self._llm.decode(
                            [biggest_logit_id])
                        prompt_output += choosed_token
                        parameter_output += choosed_token
                        print("\n" * 50 + self.terminal_output + prompt_output)
                        if prompt_output[-1] == "}":
                            if (value.type != "integer" and
                                    "." not in parameter_output):
                                prompt_output = prompt_output[:-1]
                                prompt_output += ".0}"
                            break
                    if i == 31 and prompt_output[-1] != "}":
                        prompt_output += "}"
            elif value.type == "boolean":
                biggest_logit_id = self.find_max_logit_id(
                    partial_prompt + prompt_output, {
                        valid_tokens_bool_true[0], valid_tokens_bool_false[0]})
                if biggest_logit_id == valid_tokens_bool_true[0]:
                    for token in valid_tokens_bool_true:
                        prompt_output += self._llm.decode([token])
                        print("\n" * 50 + self.terminal_output + prompt_output)
                else:
                    for token in valid_tokens_bool_false:
                        prompt_output += self._llm.decode([token])
                        print("\n" * 50 + self.terminal_output + prompt_output)
                if parameter != next(reversed(parameters)):
                    prompt_output += ","
                else:
                    prompt_output += "}"
                print("\n" * 50 + self.terminal_output + prompt_output)
            else:
                prompt_output += '"'
                quotation_marks = {
                    "“": "”",
                    "„": "“",
                    "«": "»",
                    "❝": "❞",
                    "「": "」",
                    "『": "』",
                    "〝": "〞",
                }
                if parameter != next(reversed(parameters)):
                    parameter_output = ""
                    valid_tokens = valid_tokens_string.copy()
                    for i in range(64):
                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output,
                            set(valid_tokens))
                        choosed_token = self._llm.decode(
                            [biggest_logit_id])
                        for left_quote, right_quote in quotation_marks.items():
                            if right_quote in choosed_token:
                                if left_quote not in parameter_output:
                                    choosed_token = '"'
                        prompt_output += choosed_token
                        parameter_output += choosed_token
                        print("\n" * 50 + self.terminal_output + prompt_output)
                        if (prompt_output[-1] == '"' or
                            (prompt_output[-1] == "," and
                             prompt_output[-2] == '"')):
                            break
                    if i == 63 and prompt_output[-1] != '"':
                        prompt_output += '"'
                    if prompt_output[-1] != ",":
                        prompt_output += ","
                        print("\n" * 50 + self.terminal_output + prompt_output)
                else:
                    parameter_output = ""
                    valid_tokens = valid_tokens_string_last.copy()
                    for i in range(64):
                        biggest_logit_id = self.find_max_logit_id(
                            partial_prompt + prompt_output,
                            set(valid_tokens))
                        choosed_token = self._llm.decode(
                            [biggest_logit_id])
                        for left_quote, right_quote in quotation_marks.items():
                            if right_quote in choosed_token:
                                if left_quote not in parameter_output:
                                    choosed_token = '"'
                        prompt_output += choosed_token
                        parameter_output += choosed_token
                        print("\n" * 50 + self.terminal_output + prompt_output)
                        if (prompt_output[-1] == '"' or
                            (prompt_output[-1] == "}" and
                             prompt_output[-2] == '"')):
                            break
                    if i == 63 and prompt_output[-1] != '"':
                        prompt_output += '"'
                    if prompt_output[-1] != "}":
                        prompt_output += "}"
                        print("\n" * 50 + self.terminal_output + prompt_output)
        return prompt_output

    def fix_invalid_json_escapes(self, json_string: str) -> str:
        result = []
        valid_escapes = '"\\/bfnrt'
        hex_digits = "0123456789abcdefABCDEF"
        inside_string = False
        for i in range(len(json_string)):
            char = json_string[i]
            if not inside_string:
                result.append(char)
                if char == '"':
                    inside_string = True
                continue
            if char == '"':
                result.append(char)
                inside_string = False
                continue
            if char == "\\":
                if i + 1 >= len(json_string):
                    result.append("\\\\")
                    continue
                next_char = json_string[i + 1]
                if next_char in valid_escapes:
                    result.append(char + next_char)
                    i += 2
                    continue
                if next_char == "u":
                    unicode_digits = json_string[i + 2:i + 6]
                    if (len(unicode_digits) == 4 and all(c in hex_digits for c
                                                         in unicode_digits)):
                        result.append(json_string[i:i + 6])
                        i += 6
                        continue
                result.append("\\\\")
                continue
            result.append(char)
        return "".join(result)

    def save_output(self, prompts_output: list[str]) -> None:
        def write_output(prompts_output: list[str]) -> str:
            output = "["
            for prompt in prompts_output:
                output += prompt
                if prompt != prompts_output[-1]:
                    output += ","
            output += "\n]"
            return output
        output = write_output(prompts_output)
        if os.path.dirname(self.output_path) != "":
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        try:
            json.loads(output)
        except json.JSONDecodeError:
            prompts_output[-1] = self.fix_invalid_json_escapes(
                prompts_output[-1])
            output = write_output(prompts_output)
            try:
                json.loads(output)
                with open(self.output_path, "w") as file:
                    file.write(output)
                return
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in {self.output_path}: {e}")
        with open(self.output_path, "w") as file:
            file.write(output)

    @staticmethod
    def print_time(initial_time: float, final_time: float, text: str) -> str:
        execution_time = final_time - initial_time
        exec_time_minutes = execution_time // 60
        exec_time_seconds = execution_time % 60
        if exec_time_minutes == 0:
            text = (f"\n{text}: {exec_time_seconds:.2f} "
                    f"{'second' if exec_time_seconds == 1 else 'seconds'}")
        else:
            text = (f"\n{text}: {exec_time_minutes:.0f} "
                    f"{'minute' if exec_time_minutes == 1 else 'minutes'} and "
                    f"{exec_time_seconds:.2f} "
                    f"{'second' if exec_time_seconds == 1 else 'seconds'}")
        return text

    def decode(self) -> None:
        prompts_output: list[str] = []
        function_ids = self.valid_tokens_functions()
        self.terminal_output += "\nCall me maybe - output generation:\n"
        for index, prompt in enumerate(self.tests_json_data):
            initial_time = time()
            functions_json_data_prompt = list(self.functions_json_data)
            partial_prompt = gen_prompt(
                functions_json_data_prompt, prompt.prompt)
            prompt_output = ("\n  {\n    \"prompt\": "
                             f"{json.dumps(prompt.prompt, ensure_ascii=False)}"
                             ",\n    \"name\": \"")
            if index == 0:
                self.terminal_output += "\n" + "=" * 130
            self.terminal_output += f"\n{index + 1}º prompt: {prompt.prompt}"
            print("\n" * 50 + self.terminal_output + prompt_output)
            partial_prompt, prompt_output = self.write_function(
                partial_prompt,
                prompt_output,
                function_ids,
                functions_json_data_prompt,
                prompt.prompt)
            prompt_output += '",\n    "parameters": {'
            print("\n" * 50 + self.terminal_output + prompt_output)
            prompt_output = self.write_parameters(partial_prompt,
                                                  prompt_output,
                                                  functions_json_data_prompt)
            prompt_output += '\n  }'
            print("\n" * 50 + self.terminal_output + prompt_output)
            final_time = time()
            self.terminal_output += prompt_output
            self.terminal_output += self.print_time(
                initial_time, final_time, "Execution time")
            print(partial_prompt)
            self.terminal_output += "\n" + "=" * 130
            print("\n" * 50 + self.terminal_output)
            prompts_output.append(prompt_output)
            self.save_output(prompts_output)
        print(f"\nFinal output saved in '{self.output_path}'")
