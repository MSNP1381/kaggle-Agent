# config_reader.py
import configparser
import os

class ConfigReader:
    def __init__(self, config_path='config.ini'):
        self.config = configparser.ConfigParser()
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        self.config.read(config_path)

    def get(self, section, key, fallback=None):
        return self.config.get(section, key, fallback=fallback)

    def getint(self, section, key, fallback=None):
        return self.config.getint(section, key, fallback=fallback)

    def getfloat(self, section, key, fallback=None):
        return self.config.getfloat(section, key, fallback=fallback)

    def getboolean(self, section, key, fallback=None):
        return self.config.getboolean(section, key, fallback=fallback)

config_reader = ConfigReader()