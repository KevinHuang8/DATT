import os
import sys
from enum import Enum
from pathlib import Path
import importlib.util
import numpy as np

thisdir = os.path.dirname(os.path.realpath(__file__))
DEFAULT_LOG_DIR = Path(thisdir) / 'logs'
DEFAULT_DATA_DIR = Path(thisdir) / 'data'
CONFIG_DIR = Path(thisdir) / 'configuration'
CONFIG_DIR.mkdir(exist_ok=True)
SAVED_POLICY_DIR = Path(thisdir) / 'saved_policies'
SAVED_POLICY_DIR.mkdir(exist_ok=True)

def import_config(config_filename):
    spec = importlib.util.spec_from_file_location("config", CONFIG_DIR / config_filename)
    
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['config'] = config_module
    spec.loader.exec_module(config_module)
    try:
        return config_module.config
    except AttributeError:
        raise ValueError(f'Config file {config_filename} must define a config object named `config`.')