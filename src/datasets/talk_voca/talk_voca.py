import json
import os
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import ListConfig
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from assets import DATASET_ROOT
from src import constants
from src.data.mesh import load_mesh
from src.datasets.base.anim import AnimBaseDataset
from src.datasets.utils_load import load_audio_features, load_images, load_tracked_data, load_vertices
from src.datasets.utils_seq import Range, avoffset_from_ms, avoffset_to_frame_delta
from src.engine.misc import filesys

from .load_swap_style import load_and_concat_all_offsets_swap, load_offsets_swap


class TalkVOCADataset(AnimBaseDataset):
    def __init__(self, config, root_talk, root_voca, csv_talk, csv_voca, is_trainset):
        csv_talk = [os.path.join(root_talk, x) for x in csv_talk]
        csv_voca = [os.path.join(root_voca, x) for x in csv_voca]
        super().__init__(config, [root_talk, root_voca], csv_talk + csv_voca, is_trainset, same_metadata=False)

        # load paired data for vocaset
        self.root_voca = root_voca
        self.real3d_speakers = config.VOCASET.REAL3D_SPEAKERS
        pairs_json = os.path.join(self.root_voca, "vocaset_data", "pairs.json")
        with open(pairs_json, "r") as fp:
            self.dtw_pairs = json.load(fp)

        clips = dict()
        for data_dict in self.data_dicts:
            if data_dict.speaker not in clips:
                clips[data_dict.speaker] = []
            clips[data_dict.speaker].append(f"{data_dict.seq_id}")
            if not data_dict.speaker.startswith("FaceTalk_"):
                print("AV Offset of {}/{} is {}ms".format(data_dict.speaker, data_dict.seq_id, data_dict.avoffset_ms))
        for spk in sorted(list(clips.keys())):
            seqs = clips[spk]
            print("style_id", f"{self.config.style_ids[spk]:2}", f"{spk:30}", f"{len(seqs):2}", seqs[:2], "...")

    def collate(self, batch):
        new_batch = []
        for i in range(0, len(batch) // 2 * 2, 2):
            j = i + 1
            # fmt: off
            i_frm_range, i_kwargs = batch[i]["tgt_frm_range"], batch[i]["load_kwargs"]
            j_frm_range, j_kwargs = batch[j]["tgt_frm_range"], batch[j]["load_kwargs"]
            i_spk, i_seq = batch[i]["speaker"], batch[i]["seq"]
            j_spk, j_seq = batch[j]["speaker"], batch[j]["seq"]
            batch[i].pop("tgt_frm_range"); batch[i].pop("load_kwargs")
            batch[j].pop("tgt_frm_range"); batch[j].pop("load_kwargs")
            # not swap
            if not self.config.gt_for_swap:
                new_batch.append(batch[i])
                new_batch.append(batch[j])
                continue

            # get swap data for validset
            if not self.is_trainset:
                swap_info = self.config.default_swap
                # swap_source = swap_info['swap']
                # if isinstance(swap_source, (list, tuple, ListConfig)):
                #     swap_source = swap_source[0]  # use the first source listed

                def _get_offsets_swap(frm_range):
                    return load_and_concat_all_offsets_swap(self.config, swap_info['swap'], swap_info, frm_range, verbose=False)
                    # return load_offsets_swap(self.config, swap_source, swap_info, frm_range, all_frames=False)

                batch[i]["offsets_swap_style"] = _get_offsets_swap(i_frm_range)
                batch[j]["offsets_swap_style"] = _get_offsets_swap(j_frm_range)

                new_batch.append(batch[i])
                new_batch.append(batch[j])
                continue

            if self.config.swap_using_self:
                batch[i]["offsets_swap_style"] = batch[i]["offsets_REAL"].clone()
                batch[j]["offsets_swap_style"] = batch[j]["offsets_REAL"].clone()
                new_batch.append(batch[i])
                new_batch.append(batch[j])
                continue

            if self.config.get('swap_for_e2e', False):
                batch[i]["offsets_swap_style"] = batch[i]["offsets_REAL"].clone()
                batch[j]["offsets_swap_style"] = batch[j]["offsets_REAL"].clone()
                new_batch.append(dict(I=batch[i], J=batch[j]))
                continue

            # same speaker
            if i_spk == j_spk:
                i_swp = batch[i]["offsets_REAL"].clone()
                j_swp = batch[j]["offsets_REAL"].clone()
            else:
                # load the pre-aligned data from another style
                i_kwargs["data_dir"] = os.path.join(i_kwargs["data_dir"], "dtw_mel")
                j_kwargs["data_dir"] = os.path.join(j_kwargs["data_dir"], "dtw_mel")
                i_swp = load_vertices(self.config, i_frm_range, sub_path=f"from-{j_spk}-{j_seq}-off.npy", **i_kwargs)
                j_swp = load_vertices(self.config, j_frm_range, sub_path=f"from-{i_spk}-{i_seq}-off.npy", **j_kwargs)
            batch[i]["offsets_swap_style"] = i_swp
            batch[j]["offsets_swap_style"] = j_swp
            # random scale
            if self.config.random_scale_offsets:
                raise NotImplementedError()
                scale_i = np.random.uniform(0.8, 1.25)
                scale_j = np.random.uniform(0.8, 1.25)
                # print("scale_i", scale_i, "scale_j", scale_j)
                batch[i]["offsets_REAL"] *= scale_i
                batch[j]["offsets_REAL"] *= scale_j
                batch[i]["offsets_swap_style"] *= scale_j
                batch[j]["offsets_swap_style"] *= scale_i
            new_batch.append(dict(I=batch[i], J=batch[j]))
            # fmt: on
        return default_collate(new_batch)

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
        # print("src", src_frm_range.stt_idx, src_frm_range.n_frames)
        # print("tgt", tgt_frm_range.stt_idx, tgt_frm_range.n_frames)

        if self.config.pad_tgt:
            tgt_frm_range = src_frm_range

        ret: Dict[str, Any] = dict(
            weight=1.0,
            idle_verts=self.speaker_idle_dict[speaker].clone(),
            clip_source=data_dict.clip_source,
            speaker=speaker,
            seq=data_dict.seq_id,
            speaker_id=self.speakers.index(speaker),
            tgt_frm_range=tgt_frm_range,
        )
        if self.config.get("style_ids") is not None:
            ret["style_id"] = self.config.style_ids[speaker]

        # * Prepare for load functions
        kwargs: Dict[str, Any] = dict(a=a, data_fps=data_fps, max_frames=data_dict.n_frames)
        kwargs["mel_hop"] = data_dict.mel_hop
        kwargs["ds_hop"] = data_dict.ds_hop

        # * ---------------------------------------------- Source data --------------------------------------------- * #

        if self.config.need_audio:
            afeat_dir = os.path.join(data_dir, "audio_features")
            assert os.path.exists(os.path.join(afeat_dir, "deepspeech.npy"))
            augmentation = self.is_trainset and (not is_generating)
            ret["audio_dict"] = load_audio_features(
                self.config,
                src_frm_range,
                data_dir=afeat_dir,
                augmentation=augmentation,
                using=list(self.config.using_audio_features),
                **kwargs,
            )

        # * ---------------------------------------------- Target data --------------------------------------------- * #

        if self.config.correct_avoffset:
            # get frame_delta (in data fps)
            avoffset = avoffset_from_ms(avoffset_ms, fps=data_fps)
            frame_delta = avoffset_to_frame_delta(avoffset)
            kwargs["frame_delta"] = frame_delta

        fitted_dir = os.path.join(os.path.dirname(data_dir), "fitted", os.path.basename(data_dir))

        # * load images
        if self.config.need_image:
            ret.update(load_images(self.config, tgt_frm_range, data_dir=data_dir, **kwargs))
            # tracked data
            ret["code_dict"] = load_tracked_data(
                self.config, tgt_frm_range, data_dir=fitted_dir, sub_path="frames", **kwargs
            )

        # * load verts

        def _load_real_offsets(tgt_frm_range):
            if data_type == "vocaset":
                assert kwargs["data_fps"] == 60
                REAL = load_vertices(self.config, tgt_frm_range, data_dir=data_dir, sub_path="offsets.npy", **kwargs)
            else:
                assert data_type in ["celebtalk", "facetalk"]
                if self.is_trainset or data_type == "celebtalk":
                    # ! For training, must load fitted data
                    REAL = load_vertices(self.config, tgt_frm_range, data_dir=fitted_dir, sub_path="meshes", **kwargs)
                    REAL = REAL - ret["idle_verts"][None, ...]
                else:
                    # ! For validation FaceTalk, we can load the REAL 3d to compute metrics
                    # print("Load real data for FaceTalk Validation !!!")
                    assert not self.is_trainset and data_type == "facetalk"
                    seq_name = os.path.basename(data_dir)
                    spk_name = os.path.basename(os.path.dirname(data_dir))
                    real_data_dir = os.path.join(DATASET_ROOT, "flame_mesh", "vocaset_data", spk_name, seq_name)
                    REAL = load_vertices(
                        self.config,
                        tgt_frm_range,
                        a=a,
                        data_fps=60,
                        frame_delta=avoffset_to_frame_delta(avoffset_from_ms(avoffset_ms, fps=60)),
                        data_dir=real_data_dir,
                        sub_path="offsets.npy",
                        max_frames=data_dict.n_frames,
                    )
            return REAL

        if self.config.need_verts:
            ret["offsets_REAL"] = _load_real_offsets(tgt_frm_range)

            if constants.GENERATING and self.config.gt_for_swap:
                if constants.KWARGS.get("swap_speaker") is None:
                    if speaker.startswith("FaceTalk"):
                        other_speakers = [x for x in self.speakers if x != speaker and x.startswith("FaceTalk")]
                        constants.KWARGS["swap_speaker"] = np.random.choice(other_speakers)
                    else:
                        constants.KWARGS["swap_speaker"] = speaker

                swp_frm_range = tgt_frm_range.copy()

                # random frm_range
                if constants.KWARGS.get("swap_speaker_range", "seq") == "rnd_frm":
                    max_idx = data_dict.n_frames - swp_frm_range.n_frames - 10
                    constants.KWARGS["swap_speaker_range"] = int(np.random.randint(10, max_idx))

                if constants.KWARGS.get("swap_speaker_range", "seq") != "seq":
                    swp_idx = constants.KWARGS.get("swap_speaker_range", "seq")
                    assert isinstance(swp_idx, int), f"Invalid swap_speaker_range {swp_idx}"
                    swp_frm_range.stt_idx = swp_idx

                i_spk, j_spk = speaker, constants.KWARGS["swap_speaker"]
                if j_spk == i_spk:
                    swp_off = _load_real_offsets(swp_frm_range)
                    ret["offsets_swap_style"] = swp_off
                    ret["idle_verts_swp"] = ret["idle_verts"].clone()
                else:
                    j_seq = None
                    for i_sent in range(40):
                        tag0 = f"{speaker}-{data_dict.seq_id},{constants.KWARGS['swap_speaker']}-sentence{i_sent+1:02d}"
                        tag1 = f"{constants.KWARGS['swap_speaker']}-sentence{i_sent+1:02d},{speaker}-{data_dict.seq_id}"
                        if tag0 in self.dtw_pairs or tag1 in self.dtw_pairs:
                            j_seq = f"sentence{i_sent+1:02d}"
                            break
                    assert j_seq is not None

                    off_npy = f"dtw_mel/from-{j_spk}-{j_seq}-off.npy"
                    ret["idle_verts_swp"] = self.speaker_idle_dict[j_spk].clone()
                    ret["offsets_swap_style"] = load_vertices(
                        self.config, swp_frm_range, data_dir=data_dir, sub_path=off_npy, **kwargs
                    )

        kwargs["data_dir"] = data_dir
        ret["load_kwargs"] = kwargs

        return ret
