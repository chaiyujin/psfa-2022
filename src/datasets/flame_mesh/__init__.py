from omegaconf import DictConfig

from .flame_mesh import FlameMeshDataset


def entrance(hparams: DictConfig, method: str):
    from .prepare import prepare, prepare_dtw_align

    if method == "prepare_vocaset":
        prepare(hparams, "vocaset")
    elif method == "prepare_coma":
        prepare(hparams, "coma")
    elif method == "prepare_dtw":
        prepare_dtw_align(hparams, "vocaset")
    else:
        raise NotImplementedError("Unknown method: {}".format(method))


__all__ = ["FlameMeshDataset", "entrance"]
