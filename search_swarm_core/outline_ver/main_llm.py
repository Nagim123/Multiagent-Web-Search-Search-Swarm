from outlines import models, generate

from typing import List

import json

def get_json_schema(query_count: int) -> str:
    schema = {
        "title": "Queries",
        "type": "object",
        "properties": {},
        "required": []
    }
    for i in range(query_count):
        schema["properties"][f"query_{i}"] = {"type": "string"}
        schema["required"].append(f"query_{i}")
    return json.dumps(schema)

class MainLLM:

    def __init__(self, model: models.Transformers, k: int = 3) -> None:
        self.querying_prompt = """
        <|user|>
        The user will provide you with instructions on what they want to find on the e-commerce website.
        Please offer them a list of search queries that they could use to find the right product.
        Queries should be short and take into account the specifics of search engines such as Amazon search.
        """
        self.model = model
        self.generator = generate.json(self.model, get_json_schema(k))


    def get_queries(self, instruction: str) -> List[str]:
        data = self.generator(self.querying_prompt + "\n\n" + instruction + "\n<|assistant|>\n")
        return [data[q] for q in data]