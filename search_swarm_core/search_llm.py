import httpx
from typing import List, Dict, Type, Literal
from dataclasses import dataclass

from pydantic import BaseModel, create_model
from openai import OpenAI

from config_reader import ConfigReader

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

def create_suitable_products(product_ids: List[str]) -> Type[BaseModel]:
    model = create_model("SuitableProducts",
        product_ids = (List[Literal[tuple(product_ids)]], ...))
    return model

class SearchLLM:

    def __init__(self, k: int) -> None:
        self.k = k
        self.selecting_prompt = f"""
        The user will provide you with a list of products that he has found using the search engine, and instructions on which product he wants to find.
        Please select the {k} products that are most suitable according to the instructions.
        Note that some properties, such as color or size, can be changed using selectable attributes, but if the product does not have size or color from instruction, do not choose it.
        Do not choose products for children unless it is explicitly stated in the instructions.
        Do not choose products that offer more than the user needs.
        Type their product IDs in the output and sort them by how well they meet the requirements. 
        """
        if ConfigReader.instance.get("proxy") == "-":
            self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))
        else:
            self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"), http_client=httpx.Client(proxy=ConfigReader.instance.get("proxy")))

    def get_candidates(self, instruction: str, search_results: List[Product]) -> List[Product]:
        user_message = "INSTRUCTION:\n" + instruction + "\n" + "PRODUCTS:\n"
        product_ids = []
        product_index = {}
        for i, product in enumerate(search_results):
            product_json = product.to_json()
            product_ids.append(f"ID{str(i)}")
            product_index[product_ids[-1]] = product
            user_message += str(product_json) + "\n"

        data = self.client.beta.chat.completions.parse(
            model=ConfigReader.instance.get("gpt_model"),
            messages=[
                {"role": "system", "content": self.selecting_prompt},
                {"role": "user", "content": user_message},
            ],
            seed=42,
            response_format=create_suitable_products(product_ids),
        )
        result = data.choices[0].message.parsed.product_ids
        return [product_index[pid] for pid in result[:self.k]] if len(result) > 0 else search_results[:self.k]