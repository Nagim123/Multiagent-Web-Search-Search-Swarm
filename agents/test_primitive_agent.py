"""
This file contains the primitive agent for testing purposes.
"""

from typing import Any, Dict
from agents.base_agent import BaseAgent

class PrimitiveAgent(BaseAgent):
    """
    Primitive agent for testing purposes.
    """
    def act(self, observation: str, valid_actions: Dict[str, Any]) -> str:
        clickables = valid_actions["clickables"]

        if "Search" in clickables:
            return "search[red jacket]"
        if "Buy Now" in clickables:
            return "click[Description]"
        
        return f"click[{clickables[2]}]"
    
    def stop(self) -> None:
        pass