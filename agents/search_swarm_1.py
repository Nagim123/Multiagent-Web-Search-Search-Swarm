from enum import Enum
from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from search_swarm_core.main_llm import MainLLM
from search_swarm_core.search_llm import SearchLLM
from search_swarm_core.search_llm import Product
from search_swarm_core.critique_llm import CritiqueLLM
from search_swarm_core.attribute_chooser_llm import AttributeChooserLLM

class Phase:
    PLANNING = "planning"
    DATA_COLLECTION = "data_collection"
    FINAL_DECISION = "final_decision"

# 104 episodes per search
class AgentAction:
    def __init__(self, action_name: str, payload: str, is_product_open: bool = False) -> None:
        self.action_name = action_name
        self.payload = payload
        self.is_product_open = is_product_open

    def __str__(self) -> str:
        return f"{self.action_name}[{self.payload}]"

class SearchSwarm(BaseAgent):

    def __init__(self) -> None:
        super().__init__()
        # self.main_llm = MainLLM()
        # self.critique_llm = CritiqueLLM()
        # self.attribute_chooser_llm = AttributeChooserLLM()
        # self.search_llms: List[SearchLLM] = []

        self.requirements = ""
        self.search_quries = []

        self.phase = Phase.PLANNING
        self.action_stack: List[AgentAction] = []
        self.current_trajectory = []
        self.prev_action = ""
        self.product_batches: List[List[Product]] = []
        
    
    def act(self, observation: str, valid_actions: Dict[str, Any]) -> str:
        text_data = observation.split(" [SEP] ")
        clickables = valid_actions["clickables"]

        if self.phase == Phase.PLANNING:
            self.requirements = ""#self.main_llm.generate_requirements(text_data[2])
            self.search_queries = ["red jacket", "blue toothbrush", "chair"]#self.main_llm.get_queries(text_data[2])
            self.phase = Phase.DATA_COLLECTION
            for query in self.search_queries:
                self.action_stack.append(AgentAction("click", "Back to Search"))
                self.action_stack.append(AgentAction("search", query))
        elif self.phase == Phase.DATA_COLLECTION:
            if self.prev_action.action_name == "search":
                self.current_trajectory = [self.prev_action]
                self.product_batches.append([])
                for clickable in clickables[2:]:
                    self.action_stack.append(AgentAction("click", clickable, is_product_open=True))
            elif self.prev_action.payload == "Description":
                self.product_batches[-1][-1].description = text_data[4] if 4 < len(text_data) else ""
                self.action_stack.append(AgentAction("click", "Features"))
                self.action_stack.append(AgentAction("click", "< Prev"))
            elif self.prev_action.payload == "Features":
                self.product_batches[-1][-1].features = text_data[4:]
                self.action_stack.append(AgentAction("click", "Reviews"))
                self.action_stack.append(AgentAction("click", "< Prev"))
            elif self.prev_action.payload == "Reviews":
                self.product_batches[-1][-1].reviews = text_data[4] if 4 < len(text_data) else ""
                self.action_stack.append(AgentAction("click", "Attributes"))
                self.action_stack.append(AgentAction("click", "< Prev"))
            elif self.prev_action.payload == "Attributes":
                self.product_batches[-1][-1].attributes = text_data[4:]
                self.action_stack.append(AgentAction("click", "< Prev"))
                self.action_stack.append(AgentAction("click", "< Prev"))
            elif self.prev_action.is_product_open:
                self.current_trajectory.append(self.prev_action)
                mutable_attributes = {}
                last_key = None
                dict_values = set(clickables[7:])
                for text_part in text_data[4:-8]:
                    if text_part not in dict_values:
                        mutable_attributes[text_part] = []
                        last_key = text_part
                    else:
                        if last_key is None:
                            raise Exception("Error while parsing product's mutable attributes")
                        mutable_attributes[last_key].append(text_part)

                self.product_batches[-1].append(Product(
                    self.prev_action.payload, text_data[-8], "", "", "", "", 
                    self.current_trajectory, mutable_attributes))
                self.current_trajectory = [self.current_trajectory[0]]
                self.action_stack.append(AgentAction("click", "Description"))
            if len(self.action_stack) == 0:
                self.phase = Phase.FINAL_DECISION
        
        if self.phase == Phase.FINAL_DECISION and len(self.action_stack) == 0:
            self.action_stack.append(AgentAction("click", "Buy Now"))
            best_candidates: List[Product] = []
            for candidates in self.product_batches:
                # Find the appropriate candidates
                best_candidates.append(candidates[0])
            top_candidate = best_candidates[0]
            attribute_to_select = []
            for attr in attribute_to_select:
                self.action_stack.append(AgentAction("click", attr))
            for a_traj in top_candidate.trajectory[::-1]:
                self.action_stack.append(a_traj)
            

        self.prev_action = self.action_stack.pop()
        return str(self.prev_action)

    def stop(self) -> None:
        print(self.product_batches)
        pass
