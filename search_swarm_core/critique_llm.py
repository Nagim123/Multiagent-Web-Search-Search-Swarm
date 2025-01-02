import httpx
from typing import List, Literal, Type
from pydantic import BaseModel, create_model
from openai import OpenAI

from config_reader import ConfigReader
from search_swarm_core.search_llm import Product


def create_best_product(product_ids: List[str]) -> Type[BaseModel]:
    model = create_model("BestProduct",
        product_id = (Literal[tuple(product_ids)], ...))
    return model

class CritiqueLLM:

    def __init__(self) -> None:
        self.critique_prompt = """
        The user will provide you a list of products they found using the search engine and the instruction which product they want to find.
        Please consider all nuances of search query and decide which product is the best to buy from the list.
        Pay attention that the product parameters such as color or size can be changed using selectable attributes.
        """
        if ConfigReader.instance.get("proxy") == "-":
            self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))
        else:
            self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"), http_client=httpx.Client(proxy=ConfigReader.instance.get("proxy")))

    def get_best_product(self, instruction: str, products: List[Product]) -> Product:
        user_message = f"INSTRUCTION: {instruction}\nPRODUCTS:\n"
        product_ids = []
        product_index = {}
        for i, product in enumerate(products):
            product_json = product.to_json()
            product_ids.append(f"ID{str(i)}")
            product_index[product_ids[-1]] = product
            user_message += str(product_json) + "\n"
        
        data = self.client.beta.chat.completions.parse(
            model=ConfigReader.instance.get("gpt_model"),
            messages=[
                {"role": "system", "content": self.critique_prompt},
                {"role": "user", "content": user_message},
            ],
            seed=42,
            response_format=create_best_product(product_ids),
        )
        choosen_product = data.choices[0].message.parsed.product_id
        return product_index[choosen_product]