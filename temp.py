from search_swarm_core.main_llm import MainLLM
from config_reader import ConfigReader

ConfigReader("config.json")

test = MainLLM()
print(test.generate_requirements("i am looking for x-large, red color women faux fur lined winter warm jacket coat, and price lower than 80.00 dollars"))