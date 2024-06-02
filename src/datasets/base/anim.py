"""
==================
 AnimBaseDataset
==================

The raw data are sequences according to videos. Each video may have several sequences.
However, sequence is too long for training. In fact, we use sub-sequences to train network.
To do so, we record each frame and sub-sequence information.
"""

import os

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset

from src.data.mesh import load_mesh
from src.datasets.utils_seq import Range, get_frm_range_from_sub_range, process_data_dicts
from src.engine.logging import get_logger
from src.engine.misc import filesys
from src.engine.misc.csv import read_csv_or_list

log = get_logger("AnimBaseDataset")


class AnimBaseDataset(Dataset):
    def __init__(self, config: DictConfig, roots, csv_sources, is_trainset, same_metadata=True):
        super().__init__()
        self.config = config
        self.is_trainset = is_trainset

        if isinstance(roots, str):
            roots = [roots]
        assert isinstance(roots, (list, tuple, ListConfig))

        if isinstance(csv_sources, str):
            csv_sources = [csv_sources]
        assert isinstance(csv_sources, (list, tuple, ListConfig))
        csv_sources = sorted([x for x in csv_sources])
        _, all_data_dicts = read_csv_or_list(csv_sources, same_metadata=same_metadata)

        # * Speakers (should be static global variables in inherited Dataset class)
        self.speakers = self.get_speakers(config)
        self.speaker_idle_dict = self.get_speaker_idle_dict(config, roots)

        # * Used sequence ids
        self.used_seq_ids = self.get_used_seq_ids(config)
        self.seq_ids_used_duration = config.get("seq_ids_used_duration", dict())

        # * The given seq_frames and pads are under wanna_fps
        self.wanna_fps = self.config.wanna_fps
        self.src_seq_frames = self.config.src_seq_frames
        self.tgt_seq_frames = self.config.tgt_seq_frames
        self.src_seq_pads = self.config.src_seq_pads

        # * All coordinates and ranges are under data's fps
        self.hop_factor = self.config.training_hop_factor if is_trainset else 1.0
        self.data_dicts, self.frm_coords, self.seq_ranges, self.sub_ranges = process_data_dicts(
            all_data_dicts,
            self.speakers,
            self.used_seq_ids,
            self.seq_ids_used_duration,
            self.seq_duration,
            self.hop_duration,
        )

        # * Log seq ids
        all_seq_ids = set()
        for d in self.data_dicts:
            all_seq_ids.add(f"{d.speaker}/{d.seq_id}")
        log.info(f"Found {len(all_seq_ids)} sequences, for {'training' if is_trainset else 'validation'}")

        # * For data augmentation
        self.random_shift_alpha = self.config.random_shift_alpha
        self.random_shift_index = self.config.random_shift_index
        if isinstance(self.random_shift_alpha, int):
            self.random_shift_alpha = float(self.random_shift_alpha)
        assert isinstance(self.random_shift_alpha, (bool, float)), f"Invalid type: '{type(self.random_shift_alpha)}'"
        assert isinstance(self.random_shift_index, bool), "Invalid type: '{}'".format(type(self.random_shift_index))

        # * Check methods are overrided by inherited class
        cls = self.__class__
        for method in ("get_data",):
            assert method in cls.__dict__, f"'{cls.__name__}' doesn't override method '{method}'!"

    @property
    def seq_duration(self) -> float:
        return self.tgt_seq_frames / self.wanna_fps

    @property
    def hop_duration(self) -> float:
        return self.seq_duration / self.hop_factor

    def __len__(self):
        return len(self.sub_ranges)

    def __getitem__(self, i_item):
        # * Get frame range (in sequence internal index) from sub-sequence range (in global index)
        sidx, frm_range = get_frm_range_from_sub_range(self.sub_ranges[i_item], self.frm_coords)

        # * Random shift time or not
        a = self._get_shift_alpha() if self.is_trainset else 0.0

        # * Random shift index
        shift = self._get_shift_index(sidx, frm_range.stt_idx, frm_range.n_frames) if self.is_trainset else 0
        frm_range.stt_idx += shift

        return self.get_data(sidx, frm_range, a=a)

    # * Should be override by inherit class!
    def get_data(self, sidx: int, frm_range: Range, a: float):
        raise NotImplementedError()

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                           Augmentation for Sequence                                          * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def _get_shift_alpha(self) -> float:
        if isinstance(self.random_shift_alpha, bool):
            # * Shift from 0 ~ 1
            return np.random.uniform(0, 1) if self.random_shift_alpha else 0.0
        elif isinstance(self.random_shift_alpha, (float, int)):
            # * Shift by given amount, either (0 ~ amount) or (1-amount ~ 1)
            amount = self.random_shift_alpha
            chance = np.random.uniform(0, 1)
            return np.random.uniform(0, amount) if chance <= 0.5 else np.random.uniform(1.0 - amount, 1.0)
        else:
            raise NotImplementedError()

    def _get_shift_index(self, sidx, stt_idx, n_frames) -> int:
        if not self.random_shift_index:
            return 0

        # * Get hop frames and max frames under data's fps
        max_frames = self.data_dicts[sidx].n_frames
        hop_frames = int(self.hop_duration * self.data_dicts[sidx].fps)
        # * If hop frames is small, we should just return 0
        if hop_frames <= 1:
            return 0

        # * Get the max amount to shift forward and backward
        shift_forward = min(hop_frames - 1, max_frames - 1 - stt_idx - n_frames)
        shift_backward = max(1 - hop_frames, 0 - stt_idx)
        if shift_backward < 0 < shift_forward:
            shift_idx = np.random.randint(shift_backward, shift_forward + 1)
        else:
            shift_idx = 0
        return shift_idx

    def collate(self, batch):
        return default_collate(batch)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                  Classmethods for loading static information                                 * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    @classmethod
    def get_speakers(cls, config):
        g_key = "g_speakers"
        if config.get("data_src") is not None:
            g_key += f"_{config.data_src}"
        # * Cache speakers in cls (if inherited, cls is child class)
        if getattr(cls, g_key, None) is None:
            assert config.get("speakers") is not None and len(config.speakers) > 0, "Empty 'speakers'"
            setattr(cls, g_key, sorted(list(config.speakers)))
        return getattr(cls, g_key)

    @classmethod
    def get_speaker_idle_dict(cls, config, roots):
        g_key = "g_speaker_idle_dict"
        if config.get("data_src") is not None:
            g_key += f"_{config.data_src}"
        # * Cache speakers's idle template vertices in cls (if inherited, cls is child class)
        if getattr(cls, g_key, None) is None:
            # * Get speakers first
            speakers = cls.get_speakers(config)
            speaker_idle_dict = dict()
            # * Iter speakers
            for spk in speakers:
                # * find speaker's root
                spk_root = None
                for root in roots:
                    for subdir in filesys.find_dirs(root, r".*", False):
                        if os.path.exists(os.path.join(subdir, spk)):
                            spk_root = os.path.join(subdir, spk)
                            break
                    if spk_root is not None:
                        break
                assert spk_root is not None, "Failed to find spk_root for {}, in {}".format(spk, root)
                # * Get the idle
                verts = None
                possible_idle_paths = [
                    os.path.join(spk_root, "identity.obj"),
                    os.path.join(spk_root, "identity", "identity.obj"),
                    os.path.join(spk_root, "fitted", "identity", "identity.obj"),
                ]
                for fpath in possible_idle_paths:
                    if not os.path.exists(fpath):
                        continue
                    verts, _, _ = load_mesh(fpath)
                    verts = torch.tensor(verts.astype(np.float32))
                    break
                assert verts is not None, "Failed to find idle template for speaker '{}'".format(spk)
                shape = tuple(verts.shape)
                assert len(shape) == 2 and shape[-1] == 3, "Invalid shape {} of loaded idle at {}".format(shape, fpath)
                # * Set idle
                speaker_idle_dict[spk] = verts
            setattr(cls, g_key, speaker_idle_dict)
        return getattr(cls, g_key)

    @classmethod
    def get_used_seq_ids(cls, config):
        used_seq_ids = config.get("used_seq_ids")
        if used_seq_ids is None or len(used_seq_ids) == 0:
            used_seq_ids = None
        return used_seq_ids
