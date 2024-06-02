import json
import os
from typing import Any, Dict

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate

from src import constants
from src.data.viseme import phoneme_to_id, phonemes_to_visemes, viseme_to_id
from src.datasets.base.anim import AnimBaseDataset
from src.datasets.utils_load import load_audio_features, load_labels, load_vertices
from src.datasets.utils_seq import Range, avoffset_from_ms, avoffset_to_frame_delta


class FlameMeshDataset(AnimBaseDataset):
    def __init__(self, config, root, csv_source, is_trainset):
        full_sources = []
        for x in csv_source:
            x = os.path.join(root, x)
            full_sources.append(x)
        super().__init__(config, [root], full_sources, is_trainset, same_metadata=False)

        self.root = root
        self.is_coma = any(x.find("coma") >= 0 for x in csv_source)

    def collate(self, batch):
        return default_collate(batch)

    def get_data(self, sidx: int, frm_range: Range, a: float = 0, is_generating: bool = False):
        # * Data information
        data_dict = self.data_dicts[sidx]
        data_type = data_dict.type.lower()
        data_dir = data_dict.data_dir
        data_fps = data_dict.fps
        speaker = data_dict.speaker
        avoffset_ms = data_dict.avoffset_ms

        # * First convert input frame range into data's fps
        frm_range = frm_range.convert_to(data_dict.fps)
        # * Get target frame range, under wanna fps
        tgt_frm_range = frm_range.convert_to(self.wanna_fps)
        # * Get source frame range, under wanna fps
        src_frm_range = tgt_frm_range.copy()
        src_frm_range.stt_idx -= self.src_seq_pads[0]
        src_frm_range.n_frames += sum(self.src_seq_pads)

        ret: Dict[str, Any] = dict(
            weight=1.0,
            idle_verts=self.speaker_idle_dict[speaker].clone(),
            clip_source=data_dict.clip_source,
            speaker=speaker,
            seq=data_dict.seq_id,
            speaker_id=self.speakers.index(speaker),
        )
        if self.config.get("style_ids") is not None:
            ret["style_id"] = self.config.style_ids[speaker]

        # * Prepare for load functions
        kwargs: Dict[str, Any] = dict(a=a, data_fps=data_fps, max_frames=data_dict.n_frames)
        kwargs["mel_hop"] = data_dict.mel_hop
        kwargs["ds_hop"] = data_dict.ds_hop

        # * ---------------------------------------------- Source data --------------------------------------------- * #

        if self.config.need_audio and not self.is_coma:
            afeat_dir = os.path.join(data_dir, "audio_features")
            assert os.path.exists(os.path.join(afeat_dir, "deepspeech.npy"))
            augmentation = self.is_trainset and (not is_generating)
            ret["audio_dict"] = load_audio_features(
                self.config, src_frm_range, data_dir=afeat_dir, augmentation=augmentation, **kwargs
            )

        # * ---------------------------------------------- Target data --------------------------------------------- * #

        fitted_dir = os.path.join(os.path.dirname(data_dir), "fitted", os.path.basename(data_dir))

        # get frame_delta (in data fps)
        avoffset = avoffset_from_ms(avoffset_ms, fps=data_fps)
        frame_delta = avoffset_to_frame_delta(avoffset)
        kwargs["frame_delta"] = frame_delta

        def _load_real_offsets(tgt_frm_range):
            if data_type in ["celebtalk", "facetalk"]:
                REAL = load_vertices(self.config, tgt_frm_range, data_dir=fitted_dir, sub_path="meshes", **kwargs)
                REAL = REAL - ret["idle_verts"][None, ...]
            else:
                # vocaset or coma
                REAL = load_vertices(self.config, tgt_frm_range, data_dir=data_dir, sub_path="offsets.npy", **kwargs)
            return REAL

        # * load verts
        if self.config.need_verts:
            ret["offsets_REAL"] = _load_real_offsets(tgt_frm_range)

        # * load label
        if self.config.need_label:
            phonemes = load_labels(self.config, tgt_frm_range, a=a, data_dir=data_dir, sub_path="phonemes.lab")
            if phonemes is not None:
                phoneme_ids = [phoneme_to_id(x) for x in phonemes]
                visemes = phonemes_to_visemes(phonemes)
                viseme_ids = [viseme_to_id(x) for x in visemes]
                ret["viseme_ids"] = torch.tensor(viseme_ids, dtype=torch.long)
                ret["phoneme_ids"] = torch.tensor(phoneme_ids, dtype=torch.long)

        return ret
