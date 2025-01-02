from outlines import models, generate

from typing import List

from search_swarm_core.outline_ver.search_llm import Product

import json

def get_json_schema(min_id: int, max_id: int) -> str:
    schema = {
        "title": "Best product",
        "type": "object",
        "properties": {
            "reason": {"type": "string"},
            "best_product_id": {"type": "integer", "minimum": min_id, "maximum": max_id}
        },
        "required": ["reason", "best_product_id"]
    }
    return json.dumps(schema)

class CritiqueLLM:

    def __init__(self, model: models.Transformers) -> None:
        self.critique_prompt = """
        <|user|>
        The user will provide you a list of products they found using the search engine and the instruction which product they want to find.
        Please consider all nuances of search query and decide which product is the best to buy from the list.
        Pay attention that the product parameters such as color or size can be changed using selectable attributes.
        """
        self.model = model

    def get_best_product(self, instruction: str, products: List[Product]) -> Product:
        user_message = f"INSTRUCTION: {instruction}\nPRODUCTS:\n"
        for i, product in enumerate(products):
            product_json = product.to_json()
            product_json["product_id"] = i + 1
            user_message += str(product_json) + "\n"
        
        generator = generate.json(self.model, get_json_schema(1, len(products)))
        data = generator(self.critique_prompt + "\n\n" + user_message + "\n<|assistant|>\n")
        choosen_product = data["best_product_id"]
        return products[choosen_product - 1]