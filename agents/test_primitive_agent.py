"""
This file contains the primitive agent for testing purposes.
"""

from typing import Any, Dict
from agents.base_agent import BaseAgent

class PrimitiveAgent(BaseAgent):
    """
    Primitive agent for testing purposes.
    """
    def __init__(self):
        super().__init__()
        self.temp_lock = 0

    def act(self, observation: str, valid_actions: Dict[str, Any]) -> str:
        clickables = valid_actions["clickables"]
        print(clickables)
        print(observation)

        if "search" in clickables:
            return "search[red jacket]"
        if "buy now" in clickables:
            if self.temp_lock == 0:
                self.temp_lock = 1
                return "click[features]"
            elif self.temp_lock == 1:
                self.temp_lock = 2
                return "click[buy now]"
        elif self.temp_lock > 0:
            return "click[< prev]"
        
        return f"click[{clickables[3]}]"
    
    def stop(self) -> None:
        pass