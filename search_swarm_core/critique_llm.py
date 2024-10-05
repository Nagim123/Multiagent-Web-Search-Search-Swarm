from typing import List
from pydantic import BaseModel
from openai import OpenAI

from config_reader import ConfigReader
from search_swarm_core.search_llm import Product

class BestProduct(BaseModel):
    product_id: int

class CritiqueLLM:

    def __init__(self) -> None:
        self.critique_prompt = """
        Place prompt here
        """
        self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))

    def get_best_product(self, products: List[Product], requirements: str, initial_query: str) -> Product:
        user_message = "REQUIREMENTS:\n" + "\n".join(requirements) + "\n" + "PRODUCTS:\n"
        for i, product in enumerate(products):
            product_json = product.to_json()
            product_json["product_id"] = i
            user_message += str(product_json) + "\n"
        
        data = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": self.critique_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=BestProduct,
        )
        return products[data.choices[0].message.parsed.product_id]