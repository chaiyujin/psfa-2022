import re

from omegaconf import DictConfig, ListConfig, OmegaConf


def register_custom_resolvers():
    from src.datasets.vocaset import speakers_to_string

    # register some resolvers
    OmegaConf.register_new_resolver("len", lambda x, *args: len(x))
    OmegaConf.register_new_resolver("fmt", lambda fmt, *args: fmt.format(*args))
    OmegaConf.register_new_resolver("cat", lambda *args: cat(*args))
    OmegaConf.register_new_resolver("cond", lambda x, a, b, *args: a if x else b)
    OmegaConf.register_new_resolver("case", lambda x, *args: case_of(x, *args))
    OmegaConf.register_new_resolver("join", lambda split, x, *args: split.join(str(i) for i in x))
    OmegaConf.register_new_resolver("lower", lambda x, *args: x.lower())
    OmegaConf.register_new_resolver("upper", lambda x, *args: x.upper())
    OmegaConf.register_new_resolver("rev_map", lambda x, *args: DictConfig({v: i for i, v in enumerate(x)}))
    OmegaConf.register_new_resolver("index_of", lambda list_like, x, *args: list(list_like).index(x))
    OmegaConf.register_new_resolver("vocaspk", lambda x, *args: speakers_to_string(x))


def case_of(x, *args):
    assert len(args) % 2 == 0
    if isinstance(x, (float, int)):
        for i in range(0, len(args), 2):
            case_val = args[i]
            return_val = args[i + 1]
            if x == case_val:
                return return_val
    else:
        for i in range(0, len(args), 2):
            pattern = args[i]
            value = args[i + 1]
            if re.match(pattern, x):
                return value
    raise ValueError("Failed to find any matched pattern for '{}' in '{}'".format(x, args))


def cat(*args):
    ret = []
    for x in args:
        ret += list(x)
    return ListConfig(ret)
