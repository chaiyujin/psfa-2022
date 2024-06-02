from omegaconf import DictConfig

from .talk_video import TalkVideoDataset


def entrance(hparams: DictConfig, method: str):

    if method == "prepare_celebtalk":
        from .prepare import prepare_celebtalk

        prepare_celebtalk(hparams, hparams.dataset.source_root, hparams.dataset.root)
    elif method == "prepare_facetalk":
        from .prepare import prepare_facetalk

        prepare_facetalk(hparams, hparams.dataset.source_root, hparams.dataset.root)
    else:
        raise NotImplementedError("Unknown method: {}".format(method))


__all__ = ["samplers", "read_tracked", "TrackedVideoDataset", "entrance"]
