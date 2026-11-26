import json
from pathlib import Path
from pydantic import BaseModel
from src.models import FunctionCallOutput


class OutputHandler(BaseModel):
    output_path: Path

    def save_output(self, outputs: list[FunctionCallOutput]) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [output.model_dump() for output in outputs]
        with self.output_path.open("w") as file:
            json.dump(data, file, indent=4, ensure_ascii=False,
                      allow_nan=False)
            file.write("\n")
