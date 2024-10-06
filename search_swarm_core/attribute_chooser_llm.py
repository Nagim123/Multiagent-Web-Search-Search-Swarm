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
        The user will provide you with the product that he has currently selected in the store, and instructions on which product he wants to find.
        You will also be provided with attributes that you can choose for the product.
        Please select the most appropriate parameters for each attribute so that the product conforms to the instructions provided as much as possible.
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

