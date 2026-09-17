import os
import json
import yaml
from typing import Any

class CyphaFiles:
    def read(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    def write(self, path: str, content: str):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    def parse_json(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    def parse_yaml(self, path: str) -> Any:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    def list_files(self, folder: str):
        return [os.path.join(folder, f) for f in os.listdir(folder)]

