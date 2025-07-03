
import copy
import os
import toml


class Config:
    _instance = None

    def __new__(cls, config_file=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._config = {}
            self._initialized = True
    
    def init(self, config_file=None):
        if hasattr(self, "_loaded") and self._loaded:
            return
        if config_file:
            self.load(config_file)
        else:
            default_config_path = os.path.join(os.path.dirname(__file__), "default_config.toml")
            self.load(default_config_path)
        self._loaded = True
    
    def load(self, config_file):
        if isinstance(config_file, str):
            with open(config_file, "r", encoding="utf-8") as f:
                self._config = toml.load(f)
        elif isinstance(config_file, dict):
            default_path = os.path.join(os.path.dirname(__file__), "default_config.toml")
            with open(default_path, "r", encoding="utf-8") as f:
                file_config = toml.load(f)
            self._config = self.__merge_dicts(file_config, config_file)
        else:
            raise TypeError("config_file должен быть путём к файлу или словарём")
    
    def get(self, key_path, default=None):
        keys = key_path.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def __getitem__(self, key):
        return self._config[key]
    
    def __merge_dicts(self, base, override):
        result = copy.deepcopy(base)
        for k, v in override.items():
            if (
                k in result 
                and isinstance(result[k], dict) 
                and isinstance(v, dict)
            ):
                result[k] = self.__merge_dicts(result[k], v)
            else:
                result[k] = v
        return result


def get_url(selector, datetime_point):
    config = Config()
    base_url = config.get("urls.base")
    if base_url is None:
        raise RuntimeError("Базовый URL не найден в конфигурации")
    selectors = config.get("urls.selectors")
    if selectors is None:
        raise RuntimeError("Селекторы не найдены в конфигурации")
    if selector not in selectors:
        raise ValueError(f"Unknown selector: {selector}")
    return f"{base_url}/{datetime_point}-{selectors[selector]}"
