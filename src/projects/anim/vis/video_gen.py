import json
import logging
import os
from shutil import copyfile
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ffmpeg
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm, trange

from assets import DATASET_ROOT
from src.data.mesh import load_mesh
from src.datasets.base.anim import AnimBaseDataset, Range
from src.datasets.talk_voca import load_and_concat_all_offsets_swap
from src.datasets.utils_load import load_audio_features, load_vertices
from src.datasets.utils_seq import avoffset_from_ms, avoffset_to_frame_delta, iterators
from src.engine.misc import filesys
from src.engine.train import print_dict
from src.libmorph.config import DATA_DIR as FLAME_DATA_DIR

from ..seq_info import SeqInfo
from . import painter
from .results_writer import ResultsWriter

log = logging.getLogger(__name__)


def is_audio_file(audio_path):
    _, ext = os.path.splitext(audio_path)
    return ext.lower() in [".wav", ".mp3", ".ogg", ".flac"]


def to_cmd_path(path):
    return path.replace("'", "\\'").replace(" ", "\\ ").replace('"', '\\"').replace("(", "\\(").replace(")", "\\)")


class VideoGenerator(object):
    def __init__(self, pl_module):
        self.pl_module = pl_module
        self.hparams: DictConfig = pl_module.hparams

        self.fps: float = self.hparams.wanna_fps
        self.grid_size: int = self.hparams.visualizer.video_grid_size

    @property
    def dump_audio(self):
        return self.hparams.visualizer.get("dump_audio", False)

    @property
    def video_postfix(self):
        txt = self.hparams.visualizer.get("video_postfix")
        if txt is None:
            txt = ""
        txt = txt.replace("\\", "")
        return txt

    @torch.no_grad()
    def generate_for_dataset(
        self,
        media_dir: str,
        epoch: int,
        dataset: AnimBaseDataset,
        set_type: str,
        seq_list: Union[str, List[int], Tuple[int, ...]] = "random-one",
        max_duration: Optional[float] = None,
        draw_fn_list: Tuple[Callable, ...] = (),
        extra_tag: Optional[str] = None,
        **dkwargs,  # used to overwrite data
    ):
        # * Guards
        # - no painter, just return
        if len(draw_fn_list) == 0:
            log.error("[generate_video_from_dataset]: nothing in 'draw_fn_names'!")
            return

        # * Args
        # - get max_frames
        max_frames = None
        if max_duration is None:
            max_duration = self.hparams.visualizer.get("max_duration")
        if max_duration is not None:
            max_frames = int(max_duration * self.fps)

        # * Get extra data from 'visualizer.dkwargs'
        if self.hparams.visualizer.get("dkwargs") is not None:
            for k, v in self.hparams.visualizer.dkwargs.items():
                dkwargs[k] = v

        # * Cache seq_info and change to 1
        # * We will generate frame 1 by 1
        src_seq_frames = dataset.src_seq_frames
        tgt_seq_frames = dataset.tgt_seq_frames
        # dataset.src_seq_frames = 1
        # dataset.tgt_seq_frames = 1
        # if hasattr(self.pl_module, "seq_info"):
        #     self.pl_module.seq_info._n_frames = 1
        hop_subseq = dataset.src_seq_frames

        # * Make sure run in eval mode
        self.pl_module.eval()
        # * Iter sequence
        seq_iterator = iterators.SequenceIterator(dataset, seq_list, hop_subseq)
        for seq_name, subseq_indices, data_dict in tqdm(seq_iterator, f"Video for dataset({set_type})", leave=False):
            # source of audio
            src_audio_path = os.path.join(data_dict["data_dir:path"], "audio.wav")
            # TODO: format of output prefix
            output_prefix = os.path.join(media_dir, f"[{epoch}][{set_type}]{seq_name}{self.video_postfix}")

            # (Optional) copy audio
            if self.dump_audio:
                os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
                copyfile(src_audio_path, os.path.join(output_prefix, "audio.wav"))

            # * Create writer
            results_writer = ResultsWriter(
                self.pl_module,
                out_prefix=output_prefix,
                grid_size=self.grid_size,
                fps=self.fps,
                src_audio_path=src_audio_path,
                draw_fn_list=draw_fn_list,
                draw_name_postfix=self.hparams.visualizer.get("draw_name_postfix", True),
                extra_tag=extra_tag,
            )
            # nothing to write
            if results_writer.is_empty():
                continue

            # * Loop over sub-sequences
            cnt_frames = 0
            state: Dict[str, Any] = dict()
            sub_iterator = iterators.SubSeqIterator(dataset, subseq_indices, max_frames=max_frames, **dkwargs)
            total_frames = sum(x[1].n_frames for x in subseq_indices)
            if max_frames is None:
                max_frames = total_frames
            max_frames = min(total_frames, max_frames)

            sub_progress = trange(max_frames, leave=False)
            for i_sub, (subseq_frameids, batch) in enumerate(sub_iterator):

                if batch is None:
                    continue

                # * make sure run in eval mode
                self.pl_module.eval()
                # run one batch
                subseq_frameids, batch, results, hiddens = self.run_batch(
                    i_sub, subseq_frameids, batch, state, is_trainset=(set_type == "train")
                )
                # append to towrite
                results_writer.append(subseq_frameids, batch, results, hiddens)
                if results_writer.nothing_written:
                    break

                # maybe write
                n_new_frames = results_writer.write_history(force_all=False)
                cnt_frames += n_new_frames
                sub_progress.update(n_new_frames)
            sub_progress.close()

            # clear history and close
            results_writer.write_history(force_all=True)
            results_writer.close()

        # * set back
        dataset.src_seq_frames = src_seq_frames
        dataset.tgt_seq_frames = tgt_seq_frames
        if hasattr(self.pl_module, "seq_info"):
            self.pl_module.seq_info._n_frames = src_seq_frames

    # * Can be override by inherited class
    def run_batch(self, i_sub, subseq_frameids, batch, state, is_trainset):
        state["hiddens"] = dict()

        # * Init state at first batch
        if i_sub == 0:
            # for lstm
            state["latent_lstm"] = dict(state=dict())
            # for cached states
            state["cache"] = dict()
        assert "latent_lstm" in state and "state" in state["latent_lstm"]

        # * for auto-regressive
        n_win, n_hop = None, None
        if self.hparams.get("animnet") is not None and self.hparams.animnet.get("prev_out") is not None:
            n_win = self.hparams.animnet.prev_out.n_win
            n_hop = self.hparams.animnet.prev_out.n_hop
        if n_win is not None:
            if i_sub == 0:
                state["cached_out_delta"] = torch.zeros((1, 1, n_win * n_hop, 5023, 3), dtype=torch.float32).cuda()
            if not is_trainset:
                batch["delta_dict"]["prev_out_delta"] = state["cached_out_delta"][:, :, ::n_hop, :, :]

        self.pl_module.eval()
        if hasattr(self.pl_module, "animnet"):
            results = self.pl_module.run_animnet(batch, **state, dont_render=True)
        else:
            results = self.pl_module(batch, **state)
        hiddens = state["hiddens"]

        # * update state
        if (n_win is not None) and ("animnet" in results) and (results["animnet"].get("dfrm_delta") is not None):
            out_delta = results["animnet"]["dfrm_delta"].detach()
            prv_delta = state["cached_out_delta"]
            new_delta = torch.cat((prv_delta[:, :, 1:], out_delta.unsqueeze(2)), dim=2)
            state["cached_out_delta"] = new_delta

        return subseq_frameids, batch, results, hiddens

    @torch.no_grad()
    def generate_for_media(
        self,
        media_dir: str,
        epoch: int,
        media_list: List[str],
        max_duration: Optional[float] = None,
        draw_fn_list: Tuple[Callable, ...] = (),
        extra_tag: Optional[str] = None,
        **dkwargs,  # used to overwrite data
    ):
        # * Guard
        if len(draw_fn_list) == 0:
            log.error("[generate_video_from_dataset]: nothing in 'draw_fn_names'!")
            return

        # max frames
        max_frames = None
        if max_duration is None:
            max_duration = self.hparams.visualizer.get("max_duration")
        if max_duration is not None:
            max_frames = int(max_duration * self.fps)

        # * Get extra data from 'visualizer.dkwargs'
        if self.hparams.visualizer.get("dkwargs") is not None:
            for k, v in self.hparams.visualizer.dkwargs.items():
                dkwargs[k] = v
            print_dict("Given dkwargs", dkwargs)

        # * cache and change to 1
        src_seq_frames = 1
        if hasattr(self.pl_module, "seq_info"):
            src_seq_frames = self.pl_module.seq_info._n_frames
            # self.pl_module.seq_info._n_frames = 1

        # * get sequence iterator, slightly overlap each other
        hop_subseq = 1
        seq_info = SeqInfo(1, [0, 0])
        if hasattr(self.pl_module, "seq_info"):
            hop_subseq = self.pl_module.seq_info.n_frames_valid
            seq_info = self.pl_module.seq_info
        if hasattr(self.pl_module, "need_hop") and self.pl_module.need_hop is not None:
            hop_subseq = self.pl_module.need_hop

        log.info("MEDIA DIR: '{}'".format(os.path.abspath(media_dir)))

        # * make sure run in eval mode
        self.pl_module.eval()

        # * generate videos
        progress = tqdm(media_list, "Video for media", leave=False)
        for media in progress:
            # parse media source
            extra_kwargs = dict()
            assert isinstance(media, (tuple, list))
            tag, seq_name, media_path = media[:3]
            if len(media) >= 4:
                extra_kwargs = media[3]
            # must exist source media
            media_path = os.path.expanduser(media_path)
            if not os.path.exists(media_path):
                log.warning("Failed to find {}".format(media_path))
                continue
            progress.set_description(seq_name)

            output_prefix = os.path.join(media_dir, f"[{epoch}][test]" + tag, seq_name + self.video_postfix)
            os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

            # * get input data
            media_info = get_media_info(self.hparams, output_prefix, tag, seq_name, media_path, extra_kwargs)

            if self.dump_audio:
                os.makedirs(output_prefix, exist_ok=True)
                copyfile(media_info["audio_path"], os.path.join(output_prefix, "audio.wav"))

            # * create writer
            results_writer = ResultsWriter(
                self.pl_module,
                out_prefix=output_prefix,
                grid_size=self.grid_size,
                fps=self.fps,
                src_audio_path=media_info["audio_path"],
                draw_fn_list=draw_fn_list,
                draw_name_postfix=self.hparams.visualizer.get("draw_name_postfix", True),
                extra_tag=extra_tag,
            )
            if results_writer.is_empty():
                continue

            n_frames = media_info["n_frames"]
            if max_frames is not None:
                n_frames = min(max_frames, n_frames)

            correct_avoffset = self.hparams.data_info.correct_avoffset

            # * Loop over sub-sequences
            cnt_frames = 0
            state: Dict[str, Any] = dict()
            sub_iterator = list(range(0, n_frames, hop_subseq))
            for i_sub, stt_idx in enumerate(tqdm(sub_iterator, leave=False)):

                # * prepare batch
                frm_range = Range(stt_idx, seq_info.n_frames_valid, self.fps)
                subseq_frameids = list(iter(frm_range))
                src_frm_range = frm_range.copy()
                src_frm_range.stt_idx -= seq_info.padding[0]
                src_frm_range.n_frames += sum(seq_info.padding)

                if correct_avoffset:
                    # ! recover avoffset which is removed in training
                    removed_avoffset_ms = self.hparams.visualizer.get("avoffset_ms", 0)
                    removed_avoffset = avoffset_from_ms(removed_avoffset_ms, self.fps)
                    src_frm_range.stt_idx -= avoffset_to_frame_delta(removed_avoffset)

                # Hack
                if self.hparams.data_info.pad_tgt:
                    frm_range = src_frm_range

                data = dict(
                    speaker=media_info.get("speaker", ""),
                    clip_source=media_info.get("clip_source", ""),
                )

                # fmt: off
                if media_info.get('offsets_REAL_path') is not None:
                    json_path = os.path.join(os.path.dirname(media_info.get('offsets_REAL_path')), 'info.json')
                    with open(json_path) as fp:
                        this_info = json.load(fp)
                    data["offsets_REAL"] = load_vertices(
                        self.hparams, frm_range, a=0,
                        data_fps=this_info["fps"], max_frames=this_info["n_frames"],
                        frame_delta=avoffset_to_frame_delta(this_info["avoffset"]) if correct_avoffset else 0,
                        data_dir=os.path.dirname(media_info['offsets_REAL_path']),
                        sub_path=os.path.basename(media_info['offsets_REAL_path']),
                    )

                if media_info.get("fitted_meshes_dir") is not None:
                    if correct_avoffset and removed_avoffset_ms != 0:
                        # ! given avoffset should match with training / validation data (fitted)
                        assert removed_avoffset_ms == media_info["avoffset_ms"]
                    data["offsets_tracked"] = load_vertices(
                        self.hparams, frm_range, a=0,
                        data_fps=media_info["fps"], max_frames=media_info["n_frames"],
                        data_dir=os.path.dirname(media_info.get("fitted_meshes_dir")),
                        sub_path=os.path.basename(media_info.get("fitted_meshes_dir")),
                        frame_delta=avoffset_to_frame_delta(media_info["avoffset"]) if correct_avoffset else 0,
                    )
                    data["offsets_tracked"] -= media_info['idle_verts'][None, ...]

                if "swap" in extra_kwargs and i_sub == 0:
                    data["offsets_swap_style"] = load_and_concat_all_offsets_swap(self.hparams, extra_kwargs["swap"], extra_kwargs, frm_range)
                    print("!!!! get offsets_swap_style of length: {}".format(len(data["offsets_swap_style"])))

                data["audio_dict"] = load_audio_features(DictConfig(dict(
                    using_audio_features=['deepspeech', 'deepspeech_60fps', 'for_upper'],  # ! HACK
                    company_audio="win",  # ! HACK
                    audio=self.hparams.audio,
                    video=self.hparams.video,
                )), src_frm_range, a=0, data_dir=media_info['audio_feats_dir'], augmentation=False, ds_hop=1/50.0, mel_hop=0.008)
                # fmt: on

                # * update the idle to same template
                same_idle = extra_kwargs.get("same_idle", self.hparams.visualizer.get("same_idle", False))
                if same_idle:
                    tmpl_idle, _, _ = load_mesh(os.path.join(FLAME_DATA_DIR, "template", "TMPL.obj"))
                    data["idle_verts"] = tmpl_idle
                else:
                    data["idle_verts"] = media_info["idle_verts"]

                # update dkwargs
                data.update(dkwargs)

                def _to_cuda(batch):
                    for key in batch:
                        if torch.is_tensor(batch[key]):
                            batch[key] = batch[key].cuda()
                        elif isinstance(batch[key], dict):
                            _to_cuda(batch[key])
                    return batch

                batch = default_collate([data])
                if torch.cuda.is_available():
                    batch = _to_cuda(batch)

                # run pl module
                self.pl_module.eval()
                # run one batch
                subseq_frameids, batch, results, hiddens = self.run_batch(i_sub, subseq_frameids, batch, state, False)
                # HACK: only get last hoped frames
                if i_sub > 0 and hop_subseq == 1:
                    need_frames = min(hop_subseq, len(subseq_frameids))
                    subseq_frameids = subseq_frameids[-need_frames:]

                    def _trim_dict(dat):
                        for k in dat:
                            if torch.is_tensor(dat[k]) and dat[k].ndim >= 2 and dat[k].shape[1] > need_frames:
                                dat[k] = dat[k][:, -need_frames:]
                            elif isinstance(dat[k], dict):
                                _trim_dict(dat[k])

                    _trim_dict(batch)
                    _trim_dict(results)
                    _trim_dict(hiddens)
                # append to towrite
                results_writer.append(subseq_frameids, batch, results, hiddens)
                if results_writer.nothing_written:
                    break

                # maybe write
                cnt_frames += results_writer.write_history(force_all=False)

            # clear history and close
            results_writer.write_history(force_all=True)
            results_writer.close()

        # * set back
        if hasattr(self.pl_module, "seq_info"):
            self.pl_module.seq_info._n_frames = src_seq_frames


def _generate_inputs_for_media(hparams, audio_path, output_dir, fps):
    from src.datasets.utils_audio import export_audio_features

    # Audio features
    audio_feat_out_dir = os.path.join(output_dir, "audio_feats")
    os.makedirs(audio_feat_out_dir, exist_ok=True)
    audio_feats = export_audio_features(hparams, audio_path, audio_feat_out_dir)

    n_frames = int(audio_feats["ds_feats"].shape[0] * audio_feats["ds_hop"] * fps)
    return audio_feat_out_dir, n_frames


def get_media_info(hparams, output_prefix, tag, seq_name, media_path, extra_kwargs) -> Optional[Dict[str, Any]]:
    if not os.path.exists(media_path):
        log.warning("Failed to find media path: {}".format(media_path))
        return None

    def _get(key, def_val=None):
        return extra_kwargs.get(key, hparams.visualizer.get(key, def_val))

    media_info = dict()
    media_path = os.path.abspath(media_path)

    # * Case: from dataset
    if os.path.isdir(media_path) and media_path.startswith(DATASET_ROOT):
        assert os.path.exists(os.path.join(media_path, "audio.wav"))
        assert os.path.exists(os.path.join(media_path, "audio_features"))

        media_info["audio_path"] = os.path.join(media_path, "audio.wav")
        media_info["audio_feats_dir"] = os.path.join(media_path, "audio_features")
        media_info["speaker"] = os.path.basename(os.path.dirname(media_path))
        media_info["clip_source"] = ""
        with open(os.path.join(media_path, "info.json")) as fp:
            media_info.update(json.load(fp))
        assert "n_frames" in media_info

        # identity
        real_iden_npy = os.path.join(media_path, "identity.npy")
        fitted_iden_path = os.path.join(os.path.dirname(media_path), "fitted", "identity", "identity.obj")
        if os.path.exists(fitted_iden_path):
            media_info["idle_verts"] = load_mesh(fitted_iden_path)[0]
        elif os.path.exists(real_iden_npy):
            media_info["idle_verts"] = np.load(real_iden_npy)

        # fitted or real sequence
        fitted_meshes_dir = os.path.join(os.path.dirname(media_path), "fitted", os.path.basename(media_path), "meshes")
        if os.path.exists(fitted_meshes_dir):
            media_info["fitted_meshes_dir"] = fitted_meshes_dir

        if os.path.exists(os.path.join(media_path, "offsets.npy")):
            media_info["offsets_REAL_path"] = os.path.join(media_path, "offsets.npy")
        else:
            seq_name = os.path.basename(media_path)
            spk_name = os.path.basename(os.path.dirname(media_path))
            real_data_dir = os.path.join(DATASET_ROOT, "flame_mesh", "vocaset_data", spk_name, seq_name)
            if os.path.exists(real_data_dir):
                media_info["offsets_REAL_path"] = os.path.join(real_data_dir, "offsets.npy")

    # * Case: from audio or video file
    elif os.path.isfile(media_path):
        # get audio file
        if is_audio_file(media_path):
            media_info["audio_path"] = media_path
        else:
            media_info["audio_path"] = output_prefix + ".wav"
            (ffmpeg.input(media_path).output(media_info["audio_path"], ar=16000).overwrite_output().run(quiet=True))
        # get audio feature
        tmp_dir = os.path.join(hydra.utils.get_original_cwd(), ".snaps", "audio_features", tag, seq_name)
        cache_dir = extra_kwargs.get("feat_dir", tmp_dir)
        media_info["audio_feats_dir"], media_info["n_frames"] = _generate_inputs_for_media(
            hparams,
            media_info["audio_path"],
            output_dir=cache_dir,
            fps=hparams.wanna_fps,
        )

        speaker = _get("speaker")
        data_src = _get("data_src")
        assert speaker is not None
        if data_src in ["celebtalk", "facetalk"]:
            idle_path = os.path.join(
                DATASET_ROOT, "talk_video", data_src, "data", speaker, "fitted", "identity", "identity.obj"
            )
            idle_verts, _, _ = load_mesh(idle_path)
        elif data_src in ["vocaset", "voca"]:
            idle_path = os.path.join(DATASET_ROOT, "flame_mesh", "data_vocaset", speaker, "identity.obj")
            idle_verts, _, _ = load_mesh(idle_path)
        else:
            raise NotImplementedError("unknown data_src {}".format(data_src))
        media_info["speaker"] = speaker
        media_info["clip_source"] = _get("clip_source", "")
        media_info["idle_verts"] = idle_verts

    else:
        return None

    return media_info
