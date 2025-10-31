import sys, os
from hydra import compose, initialize_config_module



def load_config(config_path: str):
    """Load a configuration file using Hydra."""
    
    cfg = compose(config_name=config_path)
    return cfg

def add_pythonpath_to_sys_path():
    if "PYTHONPATH" not in os.environ or not os.environ["PYTHONPATH"]:
        return
    sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


if __name__=="__main__":
    # read yaml config file with hydra
    add_pythonpath_to_sys_path()
    with initialize_config_module(config_module="syntra.configs"):
        cfg = load_config("syntra.yaml")