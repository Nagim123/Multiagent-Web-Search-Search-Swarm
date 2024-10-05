from typing import List
from pydantic import BaseModel
from openai import OpenAI

from config_reader import ConfigReader
from search_swarm_core.search_llm import Product

class AttributeChooserLLM:

    def __init__(self) -> None:
        self.selecting_prompt = """
        Place prompt here
        """
        self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))

    def choose_atrributes(self, product: Product, button_attributes: List[str], requirements: str) -> List[str]:
        pass