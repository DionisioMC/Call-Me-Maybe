from pydantic import BaseModel, ConfigDict


class PromptInput(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    prompt: str


class ParameterDef(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    type: str


class FunctionDef(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParameterDef]
    returns: ParameterDef
