from typing import List
from pydantic import BaseModel
from openai import OpenAI

from config_reader import ConfigReader

class SearchQueries(BaseModel):
    queries: List[str]

class FollowRequirements(BaseModel):
    requirements: List[str]

class MainLLM:

    def __init__(self) -> None:
        self.querying_prompt = """
        The user will provide you with instructions on what they want to find on the e-commerce website.
        Please offer them a list of search queries that they could use to find the right product.
        Queries should be short and take into account the specifics of search engines such as Amazon search.
        """
        self.requirements_prompt = """
        The user will provide you an instruction what they want to find on e-commerce website.
        Generate a list of requirements that the product must follow to satisfy user based on a provided instruction.
        """
        self.client = OpenAI(api_key=ConfigReader.instance.get("open_ai_api_key"))


    def get_queries(self, instruction: str) -> List[str]:
        data = self.client.beta.chat.completions.parse(
            model=ConfigReader.instance.get("gpt_model"),
            messages=[
                {"role": "system", "content": self.querying_prompt},
                {"role": "user", "content": instruction},
            ],
            response_format=SearchQueries,
        )
        return data.choices[0].message.parsed.queries

    def generate_requirements(self, instruction: str) -> List[str]:
        data = self.client.beta.chat.completions.parse(
            model=ConfigReader.instance.get("gpt_model"),
            messages=[
                {"role": "system", "content": self.requirements_prompt},
                {"role": "user", "content": instruction},
            ],
            response_format=FollowRequirements,
        )
        return data.choices[0].message.parsed.requirements