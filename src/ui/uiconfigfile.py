from configparser import ConfigParser
from pathlib import Path


class Config:
    def __init__(self,config_file=None):
        self.config=ConfigParser()
        self.config.read(config_file or Path(__file__).with_suffix('.ini'))

    def get_groq_model_options(self):
        return self.config["DEFAULT"].get("GROQ_MODEL_OPTIONS").split(", ")
    
    
