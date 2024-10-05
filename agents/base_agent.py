"""
This file contains the abstract class for any agent that can interact with webshop environment.
"""

from typing import Dict, Any

class BaseAgent:
    """
    Abstract class for any agent that can interact with webshop environment.
    """
    def act(self, observation: str, valid_actions: Dict[str, Any]) -> str:
        """
        Get an action of the agent based on the current observation and the valid actions.
        
        Args:
            observation (str): The state of the environment observed by agent represented as text.
            valid_actions (Dict[str, Any]): The dictionary containing information about actions that are valid to perform.

        Returns:
            str: The action represented as string.
        """
        raise NotImplementedError()
    
    def stop(self) -> None:
        """
        Stop all internal processes inside the agent and terminate it.
        """
        raise NotImplementedError()