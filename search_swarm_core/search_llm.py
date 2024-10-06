from typing import List, Dict
from dataclasses import dataclass

from pydantic import BaseModel
from openai import OpenAI

from config_reader import ConfigReader

@dataclass
class Product:
    unique_id: str
    name: str
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
            "description": self.description,
            "features": "; ".join(self.features),
            "attributes": "; ".join(self.attributes),
            "selectable attributes": str(self.mutable_attributes)
        }

class SuitableProducts(BaseModel):
    product_ids: List[int]

class SearchLLM:

    def __init__(self) -> None:
        K = 3
        self.selecting_prompt = f"""
        The user will provide you with a list of products that he has found using the search engine, and instructions on which product he wants to find.
        Please select the {K} products that are most suitable according to the instructions.
        Note that some properties, such as color or size, can be changed using selectable attributes, but if the product does not have size or color from instruction, do not choose it.
        Do not choose products for children unless it is explicitly stated in the instructions.
        Do not choose products that offer more than the user needs.
        Type their product IDs in the output and sort them by how well they meet the requirements. 
        """
        self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))

    def get_candidates(self, instruction: str, search_results: List[Product]) -> List[Product]:
        user_message = "INSTRUCTION:\n" + instruction + "\n" + "PRODUCTS:\n"
        for i, product in enumerate(search_results):
            product_json = product.to_json()
            product_json["product_id"] = i
            user_message += str(product_json) + "\n"

        data = self.client.beta.chat.completions.parse(
            model=ConfigReader.instance.get("gpt_model"),
            messages=[
                {"role": "system", "content": self.selecting_prompt},
                {"role": "user", "content": user_message},

            ],
            response_format=SuitableProducts,
        )

        return [search_results[p_id] for p_id in data.choices[0].message.parsed.product_ids]