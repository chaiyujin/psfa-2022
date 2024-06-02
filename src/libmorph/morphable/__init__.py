from .flame import FLAME, lbs


def build_morphable(config):
    if config.using.lower() == "flame":
        return FLAME(config.flame)
    else:
        raise NotImplementedError(f"[build_morphable]: '{config.using}' is not unknown.")
