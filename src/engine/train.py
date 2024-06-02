from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union

import torch
import torch.optim
from omegaconf import DictConfig, ListConfig, open_dict
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.engine.logging import get_logger

log = get_logger("ENGINE")


def get_tensorboard_writer(trainer: Optional[Trainer]) -> SummaryWriter:
    if trainer is None:
        return None

    exp = trainer.logger.experiment
    if isinstance(exp, (tuple, list)):
        for x in exp:
            if isinstance(x, SummaryWriter):
                return x
    elif isinstance(exp, SummaryWriter):
        return exp
    log.warning("Failed to get tensorboard writer! Check if you set it.")
    return None


def get_trainset(pl_module: LightningModule, datamodule: Optional[LightningDataModule]) -> Optional[Dataset]:
    trainset = None
    if pl_module.trainer is not None:
        trainset = pl_module.trainer.train_dataloader.dataset.datasets
    elif datamodule is not None:
        if hasattr(datamodule, "trainset"):
            trainset = datamodule.trainset
        elif hasattr(datamodule, "trainset_voca"):
            trainset = datamodule.trainset_voca
        elif hasattr(datamodule, "trainset_tracked_video"):
            trainset = datamodule.trainset_tracked_video
        else:
            raise NotImplementedError()
    return trainset


def get_validset(pl_module: LightningModule, datamodule: Optional[LightningDataModule]) -> Dataset:
    validset = None
    if (
        pl_module.trainer is not None
        and pl_module.trainer.val_dataloaders is not None
        and len(pl_module.trainer.val_dataloaders) > 0
    ):
        validset = pl_module.trainer.val_dataloaders[0].dataset
    elif datamodule is not None:
        if hasattr(datamodule, "validset"):
            validset = datamodule.validset
        elif hasattr(datamodule, "validset_voca"):
            validset = datamodule.validset_voca
        elif hasattr(datamodule, "validset_tracked_video"):
            validset = datamodule.validset_tracked_video
        else:
            raise NotImplementedError()
    return validset


def average_results(outputs: List[Any], ignore: Optional[List[str]] = None) -> Optional[Union[Dict[str, Any], Any]]:
    def _dfs_add(dst, src, allow_missing, ignore):
        for key in src:
            if ignore is not None and key in ignore:
                continue
            if isinstance(src[key], dict):
                if key not in dst and allow_missing:
                    dst[key] = dict()
                    _dfs_add(dst[key], src[key], allow_missing, ignore=ignore)
                elif key in dst:
                    _dfs_add(dst[key], src[key], allow_missing, ignore=ignore)
            else:
                if key not in dst and allow_missing:
                    dst[key] = src[key]
                elif key in dst:
                    dst[key] += src[key]

    def _dfs_div(dst, divisor, ignore):
        for key in dst:
            if ignore is not None and key in ignore:
                continue
            if isinstance(dst[key], dict):
                _dfs_div(dst[key], divisor, ignore=ignore)
            else:
                dst[key] = dst[key] / float(divisor)

    if len(outputs) == 0:
        return None

    if isinstance(outputs[0], dict):
        collect: Dict[str, Any] = dict()
        for i in range(len(outputs)):
            assert isinstance(
                outputs[i], dict
            ), "Outputs should have same type as first element 'List[Dict[str, Any]]'!"
            _dfs_add(collect, outputs[i], allow_missing=(i == 0), ignore=ignore)
        _dfs_div(collect, len(outputs), ignore=ignore)
        return collect
    elif isinstance(outputs[0], (list, tuple)):
        sum_over: Any = outputs[0]
        for i in range(1, len(outputs)):
            sum_over += outputs[i]
        return sum_over / len(outputs)
    else:
        raise TypeError("Unknown output type '{}' in ouptuts.".format(type(outputs[0])))


def epoch_logging(pl_module, outputs, log_key="items_log"):
    if len(outputs) == 0:
        return
    ignore_keys = [x for x in outputs[0].keys() if x != log_key]
    collect = average_results(outputs, ignore=ignore_keys)
    # tensorboard log scalars
    if collect is not None and collect.get(log_key) is not None:
        log: Dict[str, Any] = dict(step=pl_module.current_epoch)
        for key in collect[log_key]:
            log["epoch-" + key] = collect[log_key][key]
        pl_module.log_dict(log)


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                    Optimizers                                                    * #
# * ---------------------------------------------------------------------------------------------------------------- * #


def get_lr_scheduler(optimizer, config):
    if config is None:
        return dict(
            interval="epoch",
            frequency=10000,
            strict=True,
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda x: 1.0),
        )

    ret = dict(interval=config.get("interval", "epoch"), frequency=config.get("frequency", 1), strict=True)

    # fmt: off
    kwargs = deepcopy(config)
    with open_dict(kwargs):
        kwargs.pop("name")
        if "interval" in kwargs: kwargs.pop("interval")
        if "frequency" in kwargs: kwargs.pop("frequency")
    # fmt: on

    if config.name == "LambdaLR":
        gamma = config.get("gamma", 0.99)
        start_iter = config.get("start_iter", 0)
        min_times = config.get("min_times", 0.01)

        def _decay_lr(_iter):
            times = max(min(1, gamma ** (_iter - start_iter)), min_times)
            return times

        ret["scheduler"] = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=_decay_lr)
    else:
        ret["scheduler"] = getattr(torch.optim.lr_scheduler, config.name)(optimizer=optimizer, **kwargs)

    return ret


def get_optim_and_lrsch(config, parameters):
    kwargs = deepcopy(config)
    with open_dict(kwargs):
        if "lr_scheduler" in kwargs:
            kwargs.pop("lr_scheduler")
        kwargs.pop("name")
    # optim
    optim = getattr(torch.optim, config.name)(parameters, **kwargs)
    # lr scheduler
    lrsch = get_lr_scheduler(optim, config.get("lr_scheduler"))
    return optim, lrsch


def parse_optimizers(optimizers: OrderedDict):
    keys = list(optimizers.keys())
    optims, lrchrs = [], []
    for key in keys:
        opt, lrch = optimizers[key]
        if lrch is None:
            assert len(lrchrs) == 0 or lrchrs[-1] is None
        optims.append(opt)
        lrchrs.append(lrch)
    ret_for_pl = (optims, lrchrs) if lrchrs[0] is not None else optims
    return keys, ret_for_pl


# * ---------------------------------------------------------------------------------------------------------------- * #
# *                                                Logs and Loss Dict                                                * #
# * ---------------------------------------------------------------------------------------------------------------- * #


def reduce_dict(data, reduction="none", detach=False):
    assert reduction in ["none", "mean", "sum"]
    for k in data:
        if torch.is_tensor(data[k]):
            if reduction == "mean":
                data[k] = data[k].mean()
            elif reduction == "sum":
                data[k] = data[k].sum()
            # detach
            if detach:
                data[k] = data[k].detach()
    return data


def decorate_logs(logs_dict, training, extra_prefix=""):
    prefix = "logs" if training else "val_logs"
    logs_dict = {f"{prefix}/{extra_prefix}{k}": v for k, v in logs_dict.items()}
    return logs_dict


def sum_up_losses(loss_dict, weights, training, extra_prefix=""):
    assert isinstance(weights, (dict, int, float))

    def _weight(k):
        if isinstance(weights, dict):
            return weights[k]
        else:
            return weights

    # * sum up the losses
    # - make sure each item requires grad
    if training:
        for k, v in loss_dict.items():
            assert v.requires_grad, "{} doesn't require grad!".format(k)
    # - sum up
    loss_items = []
    for k, v in loss_dict.items():
        w = _weight(k)
        if isinstance(w, (list, tuple, ListConfig)):
            w = torch.tensor(w, dtype=v.dtype, device=v.device)
        if torch.is_tensor(w):
            while w.ndim < v.ndim:
                w = w.unsqueeze(0)
        # tqdm.write(k, v.shape, w)
        v = (v * w).mean()
        loss_items.append(v)
    loss = sum(loss_items)

    # * get the log for loss
    prefix = "loss" if training else "val_loss"
    loss_log = {f"{prefix}/{extra_prefix}{k}": v.mean().item() for k, v in loss_dict.items()}

    return loss, loss_log


def print_dict(tag, data, indent="", level="INFO"):
    getattr(log, level.lower())(pretty_dict_str(tag, data, indent))


def pretty_dict_str(tag, data, indent=""):
    ret = indent + tag + ": {\n"
    for k in sorted(list(data.keys())):
        v = data[k]
        if isinstance(v, (int, float, str)) or (torch.is_tensor(v) and v.nelement() == 1):
            ret += indent + f"  {k}: {v}\n"
        elif torch.is_tensor(v):
            ret += indent + f"  {k}: {tuple(v.shape)}\n"
        elif isinstance(v, (list, tuple, ListConfig)):
            ret += indent + f"  {k}: [...] ({len(v)})\n"
        elif not isinstance(v, (dict, DictConfig)):
            ret += indent + f"  {k}\n"
    for k in sorted(list(data.keys())):
        v = data[k]
        if isinstance(v, (dict, DictConfig)):
            if len(v) > 0:
                ret += pretty_dict_str(k, v, indent + "  ") + "\n"
            else:
                ret += indent + f"  {k}: " + "{" + "}\n"
    ret += indent + "}"
    return ret
