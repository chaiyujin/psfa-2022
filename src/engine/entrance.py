import os
import sys
import warnings
from typing import List, Optional, Sequence

import hydra
import pytorch_lightning as pl
import rich
import torch
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import InterpolationKeyError
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only

from .logging import get_logger
from .misc import filesys

log = get_logger("ENGINE")


def check_exists(config):
    # * Check exp_dir tb,checkpoints exits already
    exp_dir = config.path.exp_dir
    if not os.path.isabs(exp_dir):
        exp_dir = os.path.join(hydra.utils.get_original_cwd(), exp_dir)

    if os.path.isdir(os.path.join(exp_dir, "tb")) and os.path.isdir(os.path.join(exp_dir, "checkpoints")):
        if not config.overwrite_log:
            log.warning(
                "'path.exp_dir' is occupied, you can set 'overwrite_log=true' "
                "or delete subdir 'tb/' and 'checkpoints/' to overwrite! ({})".format(exp_dir)
            )
            quit()
        else:
            log.warning(f"'path.exp_dir' is occupied, overwrite due to 'overwrite_log' is true! ({exp_dir})")


def train(config: DictConfig) -> Optional[float]:
    check_exists(config)

    # Set seed for random number generators in pytorch, numpy and python.random
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    # Init Lightning datamodule
    datamodule: Optional[LightningDataModule] = None
    if config.get("datamodule") is not None:
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule = hydra.utils.instantiate(config.datamodule)

    # Init Lightning model
    log.info(f"Instantiating model      <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Init Lightning callbacks
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config["callbacks"].items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback   <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # Init Lightning loggers
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config["logger"].items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger     <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Init Lightning trainer
    log.info(f"Instantiating trainer    <{config.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(config.trainer, callbacks=callbacks, logger=logger, _convert_="partial")

    # Send some parameters from config to all lightning loggers
    log_hyperparameters(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Train the model
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    # # Evaluate model on test set after training
    # if not config.trainer.get("fast_dev_run"):
    #     log.info("Starting testing!")
    #     trainer.test()

    # Make sure everything closed properly
    log.info("Finalizing!")
    finish(
        config=config,
        model=model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Print path to best checkpoint
    if trainer.checkpoint_callback.monitor is not None:
        log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

    # Return metric score for hyperparameter optimization
    optimized_metric = config.get("optimized_metric")
    if optimized_metric:
        return trainer.callback_metrics[optimized_metric]


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    if "datamodule" in config:
        hparams["datamodule"] = config["datamodule"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def empty(*args, **kwargs):
    pass


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, WandbLogger):
            import wandb

            wandb.finish()


def generate(config: DictConfig) -> Optional[float]:

    # Set seed for random number generators in pytorch, numpy and python.random
    seed_everything(config.seed, workers=True)

    if config.get("load_from") is not None:
        # * Load checkpoint
        checkpoint, hparams_fpath = _check_load_from(config)
        # * Merge hparams
        config = _merge_hparams(config, hparams_fpath)
        # * Auto find a good generate dir for loaded model if not set 'generate_dir'
        gen_dir = config.path.get("generate_dir")
        if gen_dir is None:
            guess_exp_dir = os.path.dirname(hparams_fpath)
            if os.path.basename(guess_exp_dir) in ["tb", "checkpoints", "ckpts", ".hydra"]:
                guess_exp_dir = os.path.dirname(guess_exp_dir)
            gen_dir = os.path.join(guess_exp_dir, "generated")
    else:
        gen_dir = config.path.get("generate_dir")
        assert gen_dir is not None, "'path.generate_dir' is not set!"

    log.info("Generate into: '{}'".format(gen_dir))

    # * Init Lightning datamodule
    datamodule: Optional[LightningDataModule] = None
    if config.get("datamodule") is not None and (
        config.get("generate_train") is True or config.get("generate_valid") is True
    ):
        log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
        datamodule = hydra.utils.instantiate(config.datamodule)
        datamodule.setup()

    # * Init Lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    if config.get("load_from") is not None:
        # * Load the state dict
        epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        state_dict = checkpoint["state_dict"]
        # * ignore
        ignores = config.get("load_from_ignores")
        if ignores is not None:
            to_remove = []
            for key in state_dict:
                for ign in ignores:
                    if key.startswith(ign):
                        to_remove.append(key)
            state_dict = {k: v for k, v in state_dict.items() if k not in to_remove}
        # -> load
        model.load_state_dict(state_dict, strict=config.strict)
    else:
        epoch = 0
        global_step = 0

    # FIXME: config the device for model to generate
    if torch.cuda.is_available():
        model.cuda()

    if config.get("epoch") is not None:
        epoch = config.epoch

    # * Generate
    model.generate(
        epoch=epoch,
        global_step=global_step,
        generate_dir=gen_dir,
        during_training=False,
        datamodule=datamodule,
    )


def _check_load_from(config: DictConfig):
    # * load_from
    assert config.load_from is not None, "You should set 'load_from'!"
    assert os.path.exists(config.load_from), "Failed to find 'load_from': {}".format(config.load_from)

    # * Automatically find the configs for this checkpoint
    stem = os.path.splitext(os.path.basename(config.load_from))[0]
    possible_paths = [
        os.path.join(filesys.ancestor(config.load_from, 1), stem + "_hparams.yaml"),
        os.path.join(filesys.ancestor(config.load_from, 1), "hparams.yaml"),
        os.path.join(filesys.ancestor(config.load_from, 2), "hparams.yaml"),
        os.path.join(filesys.ancestor(config.load_from, 2), "tb", "hparams.yaml"),
        os.path.join(filesys.ancestor(config.load_from, 2), ".hydra", "config.yaml"),
    ]
    hparams_fpath = None
    for fpath in possible_paths:
        if os.path.exists(fpath):
            hparams_fpath = fpath
            break
    assert hparams_fpath is not None, "Failed to find hparams for load_from: '{}'".format(config.load_from)
    log.info("Found hparams for load_from at: '{}'".format(hparams_fpath))

    # * Load checkpoint
    checkpoint = torch.load(config.load_from, map_location="cpu")

    return checkpoint, hparams_fpath


def _merge_hparams(config: DictConfig, hparams_fpath: str) -> DictConfig:
    # * Load and overwrite
    hparams: DictConfig = OmegaConf.load(hparams_fpath)  # type: ignore
    assert isinstance(hparams, DictConfig)

    PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    PROJ_NAME = os.path.basename(PROJ_ROOT)

    def _check_abspath(prefix, cfg, hp):
        for k in hp:
            if isinstance(hp[k], str) and hp[k].startswith("/"):
                try:
                    if k in cfg and isinstance(cfg[k], str) and cfg[k].startswith("/"):
                        hp[k] = cfg[k]
                        # print('ignore loaded hp: {}{}'.format(prefix, k))
                except InterpolationKeyError:
                    # May fail due to invalid interpolation
                    # some interpolation can only be done after merging
                    pass

                if hp[k].find(PROJ_NAME) >= 0:
                    p = hp[k].find(PROJ_NAME)
                    hp[k] = os.path.join(os.path.dirname(PROJ_ROOT), hp[k][p:])
                    # print('replace home loaded hp: {}{} -> {}'.format(prefix, k, hp[k]))
            else:
                try:
                    if isinstance(cfg[k], DictConfig) and isinstance(hp[k], DictConfig):
                        _check_abspath(k + ".", cfg[k], hp[k])
                except InterpolationKeyError:
                    pass

    # * Ignore some keys
    # simply ignored
    top_level_ignored = (
        "model/params_total",
        "model/params_trainable",
        "model/params_not_trainable",
        "path",  # most related with current environment, just ignore
        "trainer",
        "callbacks",
        "dataset",
        "datamodule",
    )
    for k in top_level_ignored:
        if k in hparams:
            hparams.pop(k)
    if "visualizer" in hparams.model:
        hparams.model.pop("visualizer")
    if "test_media" in hparams.model:
        hparams.model.pop("test_media")
    if "loss" in hparams.model:
        hparams.model.pop("loss")
    # relative path
    _check_abspath("", config, hparams)
    # merge
    config.merge_with(hparams)

    # * Merge dotlist into config again, since some are overrided by loaded configs
    # FIXME: better method to merge
    def _remove_plus(x):
        if x.startswith("++"):
            return x[2:]
        elif x.startswith("+"):
            return x[1:]
        return x

    groups = ["load_from", "experiment", "path", "data", "dataset", "datamodule"]
    dotlist = [
        _remove_plus(x)
        for x in sys.argv[1:]
        if (x.split("=")[0] not in groups) and (not x.startswith("hydra")) and (not x.startswith("test_media"))
    ]
    config.merge_with_dotlist(dotlist)

    # FIXME: set quick flag again
    config.model.debug = config.debug

    return config


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    - forcing multi-gpu friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.fast_dev_run=True>
    if config.get("fast_dev_run"):
        log.info("Running in fast_dev_run mode! <config.fast_dev_run=True>")
        config.trainer.fast_dev_run = True

    def _force_simple_datamodule_loader(config):
        if config.get("pin_memory"):
            config.pin_memory = False
        if config.get("num_workers"):
            config.num_workers = 0
        # It's possible that datamodule has some sub datamodule configuration
        for k in config:
            if isinstance(config[k], (dict, DictConfig)):
                _force_simple_datamodule_loader(config[k])

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if "datamodule" in config:
            _force_simple_datamodule_loader(config.datamodule)

    # force multi-gpu friendly configuration if <config.trainer.accelerator=ddp>
    accelerator = config.trainer.get("accelerator")
    if accelerator in ["ddp", "ddp_spawn", "dp", "ddp2"]:
        log.info(f"Forcing ddp friendly configuration! <config.trainer.accelerator={accelerator}>")
        if "datamodule" in config:
            _force_simple_datamodule_loader(config.datamodule)

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "callbacks",
        "logger",
        "trainer",
        "dataset",
        "datamodule",
        "model",
        "path",
        "seed",
    ),
    resolve: bool = True,
    console: bool = True,
    file: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree(":gear: CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    if console:
        rich.print(tree)

    if file:
        with open(os.path.join(os.getcwd(), "config_info.txt"), "w") as fp:
            rich.print(tree, file=fp)
