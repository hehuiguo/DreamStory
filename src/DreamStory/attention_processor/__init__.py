import os
import importlib

def load_processors():
    directory = os.path.dirname(__file__)
    files = os.listdir(directory)
    modules = [f[:-3] for f in files if f.endswith("AttnProcessor.py")]

    processors = {}
    for name in modules:
        module = importlib.import_module(f"DreamStory.attention_processor.{name}")
        processors[name] = getattr(module, name)
    
    return processors

attention_processors = load_processors()

def get_attn_processor_by_name(name, **kwargs):
    if name in attention_processors.keys():
        return attention_processors[name](**kwargs)
    else:
        for k, v in attention_processors.items():
            if name.lower() in k.lower():
                Warning(f"Attention Processor {name} is not implemented yet. Use {k} instead.")
                return v(**kwargs)
        raise NotImplementedError(f"Attention Processor {name} is not implemented yet.")
