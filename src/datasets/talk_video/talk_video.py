import json
import os

import torch

from src.datasets.base.anim import AnimBaseDataset, Range
from src.datasets.utils_load import load_audio_features, load_images, load_landmarks, load_tracked_data, load_vertices
from src.datasets.utils_seq import avoffset_from_ms, avoffset_to_frame_delta
from src.engine.logging import get_logger

log = get_logger("TALK_VIDEO")


class TalkVideoDataset(AnimBaseDataset):
    g_clip_source_dict = None

    def __init__(self, config, root, csv_source, is_trainset=False):
        if isinstance(csv_source, str):
            csv_source = [csv_source]
        super().__init__(config, root, [os.path.join(root, x) for x in csv_source], is_trainset)

        # * Check compatiable fps conversion
        for data_dict in self.data_dicts:
            assert (data_dict.fps / self.wanna_fps).is_integer() or (self.wanna_fps / data_dict.fps).is_integer()

        # * Get the clip source id dict
        clip_source_set = set()
        for data_dict in self.data_dicts:
            clip_source_set.add(data_dict.clip_source)
        self.clip_source_dict = self.get_clip_source_dict(config, clip_source_set)

    def get_data(self, sidx: int, frm_range: Range, a: float = 0.0, is_generating: bool = False):
        # * Data Info
        data_dict = self.data_dicts[sidx]
        data_type = data_dict.type
        data_dir = data_dict.data_dir
        data_fps = data_dict.fps
        speaker = data_dict.speaker
        mel_hop = data_dict.mel_hop
        ds_hop = data_dict.ds_hop
        avoffset_ms = data_dict.avoffset_ms

        # * First convert input frame range into data's fps
        frm_range = frm_range.convert_to(data_dict.fps)
        # * Get target frame range, under wanna fps
        tgt_frm_range = frm_range.convert_to(self.wanna_fps)
        # * Get source frame range, under wanna fps
        src_frm_range = tgt_frm_range.copy()
        src_frm_range.stt_idx -= self.src_seq_pads[0]
        src_frm_range.n_frames += sum(self.src_seq_pads)

        # print("frame_range: {}".format(frm_range))
        # print("src_range: {}".format(src_frm_range))
        # print("tgt_range: {}".format(tgt_frm_range))

        # * Prepare sequence irrelative data
        ret = dict()
        ret["data_type"] = data_type
        ret["speaker"] = speaker
        ret["clip_source"] = data_dict.clip_source  # Used for neural renderer to determine which neural texture to use.
        ret["idle_verts"] = self.speaker_idle_dict[speaker].clone()
        # (optional) style id
        if self.config.get("style_ids") is not None:
            style_id = self.config.style_ids[speaker]
            ret["style_id"] = style_id

        # * Prepare for load functions
        kwargs = dict(a=a, data_fps=data_fps, max_frames=data_dict.n_frames)
        kwargs["mel_hop"] = mel_hop
        kwargs["ds_hop"] = ds_hop

        # * Prepare input source sequence
        if self.config.need_audio:
            augmentation = self.is_trainset and (not is_generating)
            ret["audio_dict"] = load_audio_features(
                self.config,
                src_frm_range,
                data_dir=os.path.join(data_dir, "audio_features"),
                augmentation=augmentation,
                **kwargs
            )

        # * Prepare target sequence
        avoffset = avoffset_from_ms(avoffset_ms, fps=data_fps)
        frame_delta = avoffset_to_frame_delta(avoffset)
        kwargs["frame_delta"] = frame_delta

        # Images
        if self.config.need_image:
            ret.update(load_images(self.config, tgt_frm_range, data_dir=data_dir, **kwargs))

        # Fitted
        fitted_dir = os.path.join(os.path.dirname(data_dir), "fitted", os.path.basename(data_dir))

        # load tracked data
        ret["lmks_fw75"] = load_landmarks(
            self.config, tgt_frm_range, data_dir=data_dir, sub_path="lmks_fw75.npy", **kwargs
        )
        ret["code_dict"] = load_tracked_data(
            self.config, tgt_frm_range, data_dir=fitted_dir, sub_path="frames", **kwargs
        )

        # Vertices offsets
        if self.config.need_verts:
            # load tracked verts
            ret["offsets_tracked"] = (
                load_vertices(self.config, tgt_frm_range, data_dir=fitted_dir, sub_path="meshes", **kwargs)
                - ret["idle_verts"][None, ...]
            )

            # TODO: load REAL verts
            # TODO: Refer verts

        return ret

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                      classmethods for global information                                     * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    @classmethod
    def get_clip_source_dict(cls, config, clip_source_set):
        # sort, to make sure each time it's same id embedding.
        clip_sources = sorted(list(config.possible_clip_sources))
        print(clip_sources)

        if cls.g_clip_source_dict is None:
            cls.g_clip_source_dict = {src: clip_sources.index(src) for src in sorted(list(clip_source_set))}
            # Print information
            log.info("Totally {} clip sources:".format(len(cls.g_clip_source_dict)))
            for src, _id in cls.g_clip_source_dict.items():
                log.info("  (id){}: (name){}".format(_id, src))

        for src in clip_source_set:
            assert src in cls.g_clip_source_dict, "unknown clip source: {}".format(src)

        return cls.g_clip_source_dict
