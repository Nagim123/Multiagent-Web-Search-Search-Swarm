from typing import List, Dict
from dataclasses import dataclass

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

    def __str__(self) -> str:
        return f"""ID {self.unique_id}
                Name: {self.name}
                Description: {self.description}
                Features: {self.features}
                Reviews: {self.reviews}
                Attributes: {self.attributes}"""

class SearchLLM:

    def __init__(self) -> None:
        pass

    def get_candidates(self, requirements: str, search_results: List[Product]) -> List[Product]:
        pass