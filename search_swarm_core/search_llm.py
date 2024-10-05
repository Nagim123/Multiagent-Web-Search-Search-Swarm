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

class SuitableProducts(BaseModel):
    product_ids: List[int]

class SearchLLM:

    def __init__(self) -> None:
        K = 3
        self.selecting_prompt = f"""
        The user will provide you a list of products they found using the search engine and the list of requirements for the product they want to find.
        Please choose {K} products that are most suitable according to the requirements.
        Pay attention that some properties can be changed such as color or size using selectable attributes.
        Sort them by how well they fit the requirements.
        Type their product IDs in the output. 
        """
        self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))

    def get_candidates(self, requirements: List[str], search_results: List[Product]) -> List[Product]:
        user_message = "REQUIREMENTS:\n" + "\n".join(requirements) + "\n" + "PRODUCTS:\n"
        for i, product in enumerate(search_results):
            product_json = {
                "product_id": i,
                "name": product.name,
                "description": product.description,
                "features": "; ".join(product.features),
                "attributes": "; ".join(product.attributes),
                "selectable_attributes": str(product.mutable_attributes)
            }
            user_message += str(product_json) + "\n"

        data = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": self.selecting_prompt},
                {"role": "user", "content": user_message},

            ],
            response_format=SuitableProducts,
        )

        return [search_results[p_id] for p_id in data.choices[0].message.parsed.product_ids]