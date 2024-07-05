import importlib


def get_model(cfgs):
    module_full_path=cfgs['type']
    package, module_name = module_full_path.rsplit('.', 1)
    module = importlib.import_module(f'src.sdl.{package}')
    cls_obj = getattr(module, module_name, None)
    assert cls_obj is not None, f'Class \'{module_name}\' not found.'
    try:
        inst = cls_obj(**cfgs)
    except Exception as e:
        raise Exception(f"{module_name}:{e.__repr__()}")
    return inst