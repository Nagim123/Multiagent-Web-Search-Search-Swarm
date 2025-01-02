from outlines import models, generate
from typing import Dict, List

from search_swarm_core.outline_ver.search_llm import Product
import json

def get_json_schema(selectable_attrs: Dict[str, List[str]]) -> str:
    schema = {
        "title": "Attributes",
        "type": "object",
        "properties": {},
        "required": []
    }
    for attr in selectable_attrs:
        schema["properties"][attr] = {"type": "string", "enum": selectable_attrs[attr]}
        schema["required"].append(attr)
    return json.dumps(schema)

class AttributeChooserLLM:

    def __init__(self, model: models.Transformers) -> None:
        self.selecting_prompt = """
        <|user|>
        You will be provided with attributes that you can choose for the product.
        Please select the most appropriate parameters for each attribute so that the product conforms to the instructions provided as much as possible.
        Select values for each existing attribute specified in the SELECTED ATTRIBUTES section and do not add any new attributes.
        """
        self.model = model
        
    def choose_atrributes(self, product: Product, instruction: str) -> List[str]:
        if len(product.mutable_attributes) == 0: return []
        
        user_message = f"INSTRUCTION: {instruction}\nSELECTABLE ATTRIBUTES:\n"
        for key in product.mutable_attributes:
            user_message += key + ": " + str(product.mutable_attributes[key]) + "\n"
        try:
            generator = generate.json(self.model, get_json_schema(product.mutable_attributes))
            selected_attributes = generator(self.selecting_prompt + "\n\n" + user_message + "\n<|assistant|>\n")
        except Exception as e:
            print(product)
            print(get_json_schema(product.mutable_attributes))
            raise e
        choosed_attributes = []
        for key in product.mutable_attributes:
            choosed_attributes.append(selected_attributes[key])
        return choosed_attributes