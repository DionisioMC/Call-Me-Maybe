from typing import Any
from pydantic import BaseModel, PrivateAttr
from .models import FunctionDef, PromptInput, FunctionCallOutput
from .llm_service import LLMService
from .token_manager import TokenManager
from .function_selector import FunctionSelector
from .parameter_generator import ParameterGenerator
from .output_handler import OutputHandler
from .prompt_generator import gen_prompt
from pathlib import Path


class ConstrainedDecoder(BaseModel):
    functions_json_data: list[FunctionDef]
    tests_json_data: list[PromptInput]
    output_path: str
    _llm: LLMService = PrivateAttr(default_factory=LLMService)
    _token_manager: TokenManager = PrivateAttr()
    _function_selector: FunctionSelector = PrivateAttr()
    _parameter_generator: ParameterGenerator = PrivateAttr()
    _output_handler: OutputHandler = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        self._token_manager = TokenManager.from_model(self._llm)
        self._function_selector = FunctionSelector(
            functions=self.functions_json_data)
        self._parameter_generator = ParameterGenerator(
            token_manager=self._token_manager
        )
        self._output_handler = OutputHandler(
            output_path=Path(self.output_path))

    def decode(self) -> None:
        outputs: list[FunctionCallOutput] = []
        for prompt in self.tests_json_data:
            partial_prompt = gen_prompt(
                self.functions_json_data,
                prompt.prompt
            )
            function = self._function_selector.select_function(
                self._llm,
                partial_prompt
            )
            parameters = (
                self._parameter_generator.generate_parameters(
                    self._llm, function, partial_prompt))
            outputs.append(
                FunctionCallOutput(
                    prompt=prompt.prompt,
                    name=function.name,
                    parameters=parameters
                )
            )
        self._output_handler.save_output(outputs)
