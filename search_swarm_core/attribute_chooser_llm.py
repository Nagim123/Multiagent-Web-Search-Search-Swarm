from typing import List, Dict
from pydantic import BaseModel
from openai import OpenAI

from config_reader import ConfigReader
from search_swarm_core.search_llm import Product

class ProductAttribute(BaseModel):
    attribute_name: str
    value: str

class SelectedAttributes(BaseModel):
    attributes: List[ProductAttribute]

class AttributeChooserLLM:

    def __init__(self) -> None:
        self.selecting_prompt = """
        Place prompt here
        """
        self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))

    def choose_atrributes(self, product: Product, requirements: str) -> List[str]:
        user_message = f"REQUIREMENTS: {requirements}\nSELECTABLE ATTRIBUTES:\n"
        for key in product.mutable_attributes:
            user_message += key + ": " + str(product.mutable_attributes[key]) + "\n"
        
        data = self.client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            messages=[
                {"role": "system", "content": self.selecting_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=SelectedAttributes,
        )
        selected_attributes = data.choices[0].message.parsed.attribues

        return [attr.value for attr in selected_attributes]

