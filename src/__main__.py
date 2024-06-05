import re
import sys

import dotenv
import hydra
from omegaconf import DictConfig, ListConfig
from omegaconf.errors import InterpolationKeyError

from .data.video import VideoWriter
from .engine.hydra.configs import store_configs
from .engine.resolvers import register_custom_resolvers

# * Load environment variables from `.env` file if it exists.
# * Recursively searches for `.env` in all folders starting from work dir.
dotenv.load_dotenv(override=True)

# * Store structured configs
store_configs()

# * Register custom resolvers for omegaconf
register_custom_resolvers()


def import_by_name(name):
    try:
        __import__(name)
        return sys.modules[name]
    except Exception:
        raise ImportError(name)


@hydra.main(config_path="../config/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from tqdm.contrib.logging import logging_redirect_tqdm

    from . import constants
    from .engine.entrance import extras, generate, train
    from .engine.logging import get_logger, set_default_level

    if config.debug:
        set_default_level("DEBUG")
        constants.DEBUG = True
    if config.get("default_avoffset_ms") is not None:
        constants.DEFAULT_AVOFFSET_MS = config.default_avoffset_ms
    constants.MODE = config.mode

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    extras(config)

    # get logger and redirect tqdm logging
    log = get_logger("RUN")
    with logging_redirect_tqdm():
        if config.mode == "train":
            call_init_(config)
            train(config)
        elif config.mode == "generate":
            call_init_(config)
            generate(config)
        elif re.match(r"^dataset@.+$", config.mode):
            module_name = config.dataset._module_
            m = import_by_name(module_name)
            method = config.mode.split("@")[-1]
            assert hasattr(m, "entrance"), f"Failed to find 'entrance' for module {module_name}"
            log.info(f"Processing dataset <{module_name}> by method `{method}`")
            getattr(m, "entrance")(config, method)
        else:
            log.fatal(f"mode '{config.mode}' is not implemented!")


def call_init_(config: DictConfig):
    import matplotlib

    # * Specially for Matplotlib
    if config.get("matplotlib_using") is not None:
        matplotlib.use(config.matplotlib_using)

    # * Iter all sub configs and find '_init_'
    def _dfs(config):
        try:
            if isinstance(config, DictConfig) and config.get("_init_") is not None:
                # parse function module and name
                ss = config.get("_init_").split(".")
                m_name, f_name = ".".join(ss[:-1]), ss[-1]
                m = import_by_name(m_name)
                # print("call {}.{}".format(m_name, f_name))
                # call the function
                getattr(m, f_name)(config)
            # sub configs
            if isinstance(config, DictConfig):
                for _, cfg in config.items():
                    _dfs(cfg)
            elif isinstance(config, ListConfig):
                for cfg in config:
                    _dfs(cfg)
        except InterpolationKeyError:
            pass

    # call config.utils first
    _dfs(config.utils)
    # call from top level
    _dfs(config)


if __name__ == "__main__":
    main()
