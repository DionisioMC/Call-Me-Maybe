from pydantic import BaseModel, ConfigDict


class Prompt(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    prompt: str


class ParameterDefinition(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    type: str


class FunctionDef(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    name: str
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ParameterDefinition
