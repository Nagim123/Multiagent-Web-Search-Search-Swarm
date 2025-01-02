"""
File contains an implementation of proposed agent.
"""

from typing import Any, Dict, List
from agents.base_agent import BaseAgent
from search_swarm_core.outline_ver.main_llm import MainLLM
from search_swarm_core.outline_ver.search_llm import SearchLLM
from search_swarm_core.outline_ver.search_llm import Product
from search_swarm_core.outline_ver.critique_llm import CritiqueLLM
from search_swarm_core.outline_ver.attribute_chooser_llm import AttributeChooserLLM

from outlines import models
from config_reader import ConfigReader

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

    def __init__(self, query_count: int = 2) -> None:
        super().__init__()
        core_model = models.transformers(ConfigReader.instance.get("hg_model"))
        self.query_count = query_count

        self.main_llm = MainLLM(core_model)
        self.search_llm = SearchLLM(core_model, 3)
        self.critique_llm = CritiqueLLM(core_model)
        self.attribute_chooser_llm = AttributeChooserLLM(core_model)

        self.base_instruction = ""
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
            self.base_instruction = text_data[2]
            self.search_queries = self.main_llm.get_queries(text_data[2])[:self.query_count]
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
                    self.prev_action.payload, text_data[-8], text_data[-7], "", "", "", "", 
                    self.current_trajectory, mutable_attributes))
                self.current_trajectory = [self.current_trajectory[0]]
                self.action_stack.append(AgentAction("click", "Description"))
            if len(self.action_stack) == 0:
                self.phase = Phase.FINAL_DECISION
        
        if self.phase == Phase.FINAL_DECISION and len(self.action_stack) == 0:
            self.action_stack.append(AgentAction("click", "Buy Now"))
            best_candidates: List[Product] = []
            self.__clean_product_batches()
            for candidates in self.product_batches:
                # Find the appropriate candidates
                best_candidates.extend(self.search_llm.get_candidates(self.base_instruction, candidates))
            top_candidate = self.critique_llm.get_best_product(self.base_instruction, best_candidates)
            if len(top_candidate.mutable_attributes) > 0:
                attribute_to_select = self.attribute_chooser_llm.choose_atrributes(top_candidate, self.base_instruction)
                for attr in attribute_to_select:
                    self.action_stack.append(AgentAction("click", attr))
            for a_traj in top_candidate.trajectory[::-1]:
                self.action_stack.append(a_traj)
            

        self.prev_action = self.action_stack.pop()
        if self.prev_action.payload == "Buy Now" and self.prev_action.action_name == "click":
            temp = self.prev_action
            self.reset()
            return str(temp)
        return str(self.prev_action)

    def reset(self) -> None:
        self.search_quries = []

        self.phase = Phase.PLANNING
        self.action_stack: List[AgentAction] = []
        self.current_trajectory = []
        self.prev_action = ""
        self.product_batches: List[List[Product]] = []

    def __clean_product_batches(self):
        unique_products = dict()
        for products in self.product_batches:
            for product in products:
                unique_products[product.unique_id] = product
        
        list_of_products = [unique_products[unique_id] for unique_id in unique_products]
        batch_size = len(list_of_products) // len(self.product_batches)
        if batch_size == 0:
            self.product_batches = [list_of_products]
            return
        self.product_batches = []
        
        for i in range(0, len(list_of_products), batch_size):
            self.product_batches.append(list_of_products[i:i+batch_size])

    def stop(self) -> None:
        self.reset()
        pass
