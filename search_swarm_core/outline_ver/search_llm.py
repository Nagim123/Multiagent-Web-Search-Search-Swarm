from outlines import models, generate
from typing import List, Dict
from dataclasses import dataclass

import json

@dataclass
class Product:
    unique_id: str
    name: str
    price: str
    description: str
    features: List[str]
    reviews: str
    attributes: List[str]
    trajectory: List[str]
    mutable_attributes: Dict[str, str]

    def to_json(self):
        return {
            "product_id": self.unique_id,
            "name": self.name,
            "price": self.price,
            "description": self.description,
            "features": "; ".join(self.features),
            "attributes": "; ".join(self.attributes),
            "selectable attributes": str(self.mutable_attributes)
        }

def get_json_schema(k: int, min_id: int, max_id: int) -> str:
    schema = {
        "title": "Relevant products",
        "type": "object",
        "properties": {},
        "required": []
    }
    for i in range(k):
        schema["properties"][f"product_{i}"] = {"type": "integer", "minimum": min_id, "maximum": max_id}
        schema["required"].append(f"product_{i}")
    return json.dumps(schema)

class SearchLLM:

    def __init__(self, model: models.Transformers, k: int) -> None:
        self.k = k
        self.selecting_prompt = f"""
        <|user|>
        The user will provide you with a list of products that he has found using the search engine, and instructions on which product he wants to find.
        Please select the {k} products that are most suitable according to the instructions.
        Note that some properties, such as color or size, can be changed using selectable attributes, but if the product does not have size or color from instruction, do not choose it.
        Do not choose products for children unless it is explicitly stated in the instructions.
        Do not choose products that offer more than the user needs.
        Type their product IDs in the output and sort them by how well they meet the requirements. 
        """
        self.model = model

    def get_candidates(self, instruction: str, search_results: List[Product]) -> List[Product]:
        user_message = "INSTRUCTION:\n" + instruction + "\n" + "PRODUCTS:\n"
        for i, product in enumerate(search_results):
            product_json = product.to_json()
            product_json["product_id"] = i + 1
            user_message += str(product_json) + "\n"

        generator = generate.json(self.model, get_json_schema(min(self.k, len(search_results)), 1, len(search_results)))
        data = generator(self.selecting_prompt + "\n\n" + user_message + "\n<|assistant|>\n")
        return [search_results[data[p] - 1] for p in data]