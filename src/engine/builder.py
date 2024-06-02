import os
from copy import deepcopy
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf, open_dict

from .logging import get_logger
from .ops import freeze

log = get_logger("ENGINE")


def load_or_build(
    config,
    build_fn: Callable,
    possible_names: Optional[Union[str, List[str]]] = None,
    state_dict_remove_prefix: Optional[Union[str, List[str]]] = None,
    hparams_keys_kept: Optional[List[str]] = None,
    freeze_loaded: Optional[bool] = None,
):
    load_from = config.get("load_from")

    if isinstance(load_from, (dict, DictConfig)):
        # * If load_from is a config, we should load others from it
        assert possible_names is None
        assert state_dict_remove_prefix is None
        assert hparams_keys_kept is None
        possible_names = load_from.possible_names
        state_dict_remove_prefix = load_from.get("state_dict_remove_prefix", None)
        hparams_keys_kept = load_from.get("hparams_keys_kept", None)
        # * Get from config only if not set.
        if freeze_loaded is None:
            freeze_loaded = load_from.get("freeze_loaded", False)
        else:
            # Check same
            if load_from.get("freeze_loaded") is not None and load_from.freeze_loaded != freeze_loaded:
                log.warning(
                    "'freeze_loaded' is set as {} from function args, but {} in loaded config!".format(
                        freeze_loaded, load_from.freeze_loaded
                    )
                )
        load_from = load_from.path
    elif load_from is not None:
        # * If load_from is not config, others should be given by args
        assert possible_names is not None
        # * Default for freeze_loaded is False
        if freeze_loaded is None:
            freeze_loaded = False

    # * Now it's must be a string or just None
    if isinstance(load_from, str):
        if load_from.lower() in ["", "none"]:
            load_from = None
    assert load_from is None or isinstance(load_from, str)

    # * (optional) find possible hparams and override the part in config
    name = ""
    if load_from is not None and load_from.lower() != "dont_build":
        assert os.path.exists(load_from), "Failed to find checkpoint: {}".format(load_from)
        if isinstance(possible_names, str):
            possible_names = [possible_names]
        possible_names = list(possible_names)

        hparams = None
        found_path = ""
        dir0 = os.path.dirname(load_from)
        dir1 = os.path.dirname(dir0)
        possible_paths = [
            os.path.splitext(load_from)[0] + "-hparams.yaml",
            os.path.splitext(load_from)[0] + "_hparams.yaml",
            os.path.join(dir0, "hparams.yaml"),
            os.path.join(dir1, "hparams.yaml"),
            os.path.join(dir1, "tb", "hparams.yaml"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                found_path = path
                hparams = OmegaConf.load(path)
                break
        if hparams is None:
            raise FileNotFoundError(f"Failed to find related hparams.yaml for '{load_from}'")

        def _check_has_key(hparams, name):
            if name.find(".") >= 0:
                ss = name.split(".")
                k, subk = ss[0], ".".join(ss[1:])
                if k in hparams:
                    return _check_has_key(hparams[k], subk)
                else:
                    return None
            else:
                if name in hparams:
                    return deepcopy(hparams[name])
                else:
                    return None

        # get new config
        new_config = None
        for name in possible_names:
            new_config = _check_has_key(hparams, name)
            if new_config is None and "model" in hparams:
                new_config = _check_has_key(hparams.model, name)
            # found!
            if new_config is not None:
                break

        if new_config is None:
            raise ValueError(f"We failed to find any name ({possible_names}) in '{found_path}'")

        # some keys should be kept and not override
        if hparams_keys_kept is not None:
            with open_dict(new_config):
                for key in hparams_keys_kept:
                    if key in new_config:
                        new_config.pop(key)

        # update
        config.update(new_config)

    # * build module
    if load_from is not None and load_from.lower() == "dont_build":
        ret = None
    else:
        ret = build_fn(config)
        if isinstance(ret, (list, tuple)):
            assert isinstance(ret[0], nn.Module)
            module = ret[0]
        else:
            assert isinstance(ret, nn.Module)
            module = ret

    # * (optional) load state dict
    if load_from is not None and load_from.lower() != "dont_build":
        # find the state_dict
        state_dict = torch.load(load_from, map_location="cpu")
        if "state_dict" in state_dict and "epoch" in state_dict and "global_step" in state_dict:
            state_dict = state_dict["state_dict"]

        # remove prefix
        if state_dict_remove_prefix is not None and len(state_dict_remove_prefix) > 0:
            if isinstance(state_dict_remove_prefix, str):
                state_dict_remove_prefix = [state_dict_remove_prefix]

            # add '.'
            for i in range(len(state_dict_remove_prefix)):
                if state_dict_remove_prefix[i][-1] != ".":
                    state_dict_remove_prefix[i] += "."

            def _remove_prefix(key):
                for prefix in state_dict_remove_prefix:
                    if len(prefix) > 0 and key.startswith(prefix):
                        key = key[len(prefix) :]
                        return key
                return None

            new_state_dict = dict()
            for key, val in state_dict.items():
                new_key = _remove_prefix(key)
                if new_key is not None:
                    new_state_dict[new_key] = val
            state_dict = new_state_dict

        module.load_state_dict(state_dict)
        log.info(f"Load '{name}' from: '{load_from}'")

        # (Optional) freeze
        if freeze_loaded:
            freeze(module)
            log.info(f"Freeze loaded '{name}'")

    return ret
