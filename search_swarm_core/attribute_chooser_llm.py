import httpx

from typing import List, Dict, Type, Literal
from pydantic import BaseModel, create_model
from openai import OpenAI

from config_reader import ConfigReader
from search_swarm_core.search_llm import Product

def create_attribute_model(attr_name: str, values: List[str]) -> Type[BaseModel]:    
    model = create_model(f"Product_{attr_name}",
        attribute_name = (Literal[attr_name], ...),
        value = (Literal[tuple(values)], ...))
    return model

def create_selected_attributes_class(attribute_classes: List):
    val_dict = {}
    for attr_class in attribute_classes:
        val_dict[attr_class.__name__] = (attr_class, ...)
    model = create_model("SelectedAttributes", **val_dict)
    return model

class AttributeChooserLLM:

    def __init__(self) -> None:
        self.selecting_prompt = """
        You will be provided with attributes that you can choose for the product.
        Please select the most appropriate parameters for each attribute so that the product conforms to the instructions provided as much as possible.
        Select values for each existing attribute specified in the SELECTED ATTRIBUTES section and do not add any new attributes.
        """
        if ConfigReader.instance.get("proxy") == "-":
            self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))
        else:
            self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"), http_client=httpx.Client(proxy=ConfigReader.instance.get("proxy")))

    def choose_atrributes(self, product: Product, instruction: str) -> List[str]:
        if len(product.mutable_attributes) == 0: return []
        
        user_message = f"INSTRUCTION: {instruction}\nSELECTABLE ATTRIBUTES:\n"
        attr_models = []
        for key in product.mutable_attributes:
            user_message += key + ": " + str(product.mutable_attributes[key]).replace('\n', '[SEP]').replace('"', "[INCHES]") + "\n"
            attr_models.append(create_attribute_model(key, [val.replace('\n', '[SEP]').replace('"', "[INCHES]") for val in product.mutable_attributes[key]]))

        selected_attrs_class = create_selected_attributes_class(attr_models)
        try:
            data = self.client.beta.chat.completions.parse(
                model=ConfigReader.instance.get("gpt_model"),
                messages=[
                    {"role": "system", "content": self.selecting_prompt},
                    {"role": "user", "content": user_message},
                ],
                seed=42,
                response_format=selected_attrs_class,
            )
        except Exception as e:
            print(product)
            raise e
        choosed_attributes = []
        selected_attributes = data.choices[0].message.parsed
        for key in product.mutable_attributes:
            atr = selected_attrs_class.__getattribute__(selected_attributes, f"Product_{key}")
            choosed_attributes.append(atr.value.replace('[SEP]', '\n').replace('[INCHES]', '"'))
        return choosed_attributes