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
        You will be provided with attributes that you can choose for the product.
        Please select the most appropriate parameters for each attribute so that the product conforms to the instructions provided as much as possible.
        Select values for each existing attribute specified in the SELECTED ATTRIBUTES section and do not add any new attributes.
        """
        self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))

    def choose_atrributes(self, product: Product, instruction: str) -> List[str]:
        user_message = f"INSTRUCTION: {instruction}\nSELECTABLE ATTRIBUTES:\n"
        for key in product.mutable_attributes:
            user_message += key + ": " + str(product.mutable_attributes[key]) + "\n"
        
        data = self.client.beta.chat.completions.parse(
            model=ConfigReader.instance.get("gpt_model"),
            messages=[
                {"role": "system", "content": self.selecting_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=SelectedAttributes,
        )
        selected_attributes = data.choices[0].message.parsed.attributes

        return [attr.value for attr in selected_attributes if (attr.attribute_name in product.mutable_attributes and attr.value in product.mutable_attributes[attr.attribute_name])]

