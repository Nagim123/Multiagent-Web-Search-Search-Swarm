from search_swarm_core.main_llm import MainLLM
from config_reader import ConfigReader

ConfigReader("config.json")

test = MainLLM()
print(test.generate_requirements("i am looking for blue color toothbrushes that helps to maintain my oral hygiene, and price lower than 50.00 dollars"))