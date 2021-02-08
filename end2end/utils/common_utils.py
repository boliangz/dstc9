import os
import json
from collections import defaultdict




class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_config(name="configs/config.json"):
    with open(name) as f:
        config = dotdict(json.load(f))
    return config
