import json

class ConfigReader:
    instance = None

    def __init__(self, config_filepath: str) -> None:
        if ConfigReader.instance is None:
            ConfigReader.instance = self
        else:
            raise Exception("Two instances of ConfigReader were created.")
        
        self.config_data = {}
        with open(config_filepath, "r") as config_file:
            self.config_data = json.load(config_file)
    
    def get(self, key: str) -> str:
        return self.config_data[key]