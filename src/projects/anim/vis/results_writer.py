import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from assets import get_vocaset_template_triangles
from src.data import image as imutils
from src.data.mesh.io import save_obj
from src.data.video import VideoWriter
from src.engine.logging import get_logger

log = get_logger("ResultsWriter")


class _ToWrite(object):
    def __init__(self, frame_id):
        self.frame_id = frame_id
        self.updated = False
        self.data = dict()

    def update(self, ifrm, expected_frames, possible_padding, **kwargs):
        self.updated = True
        for key, val in kwargs.items():
            assert isinstance(val, dict), f"{key} is unknown type {type(val)}"
            if key not in self.data:
                self.data[key] = dict()
            self._merge(self.data[key], val, ifrm, expected_frames, possible_padding)
        #     print_dict(key, val)
        # quit()

    def concat_to(self, key, target):
        self._concat(target, self.data[key])

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                             Merge history results                                            * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def _is_shared(self, key, val):
        return (
            (torch.is_tensor(val) and val.ndim < 2)
            or (isinstance(val, (tuple, list)) and len(val) > 0 and isinstance(val[0], str))
            or (key in ["video_id", "style_id", "idle_verts", "idle_verts_swp", "speaker", "z_style"])
            or (key.find("VIDX") >= 0)
            or (key == "refer" or key == "refer_only_ctt" or key == "refer_ctt_sty")  # HACK: reference sequence
        )

    def _merge(self, frame_dict, batch, ifrm, expected_frames, possible_padding):
        def _merge_tensor(target, tensor):
            assert tensor.shape[0] == 1, f"Only work for one batch_size, but {key}({tensor.shape}) is given"
            if tensor.shape[1] > expected_frames:
                assert tensor.shape[1] == expected_frames + sum(
                    possible_padding
                ), f"Given '{key}' has shape {tensor.shape}, should be of length {expected_frames + sum(possible_padding)}"
                ss, to = possible_padding
                to = tensor.shape[1] - to
                tensor = tensor[:, ss:to]
            # print("process {}: {}".format(key, tensor.shape))
            value = tensor if tensor.shape[1] == 1 else tensor[:, ifrm : ifrm + 1]
            # Merge value
            return torch.cat((target, value), dim=0) if target is not None else value

        for key in batch:
            # * Case 1: the tensor is shared amount all frames
            if self._is_shared(key, batch[key]):
                if key not in frame_dict:
                    frame_dict[key] = batch[key]
            # * Case 2: sub dict
            elif isinstance(batch[key], dict):
                if key not in frame_dict:
                    frame_dict[key] = dict()
                self._merge(frame_dict[key], batch[key], ifrm, expected_frames, possible_padding)
            elif isinstance(batch[key], (list, tuple)):
                if key not in frame_dict:
                    frame_dict[key] = [None for _ in range(len(batch[key]))]
                for i in range(len(batch[key])):
                    frame_dict[key][i] = _merge_tensor(frame_dict[key][i], batch[key][i])
            # * Case 3: Tensor in shape (N, L, ...)
            elif torch.is_tensor(batch[key]):
                tensor = batch[key]
                # Smaller than expected frames and not one frame
                if tensor.shape[1] != 1 and tensor.shape[1] < expected_frames:
                    continue
                frame_dict[key] = _merge_tensor(frame_dict.get(key), tensor)

    def _concat(self, batch, frame_dict):
        for key in frame_dict:
            if self._is_shared(key, frame_dict[key]):
                if key not in batch:
                    batch[key] = frame_dict[key]
            elif isinstance(frame_dict[key], dict):
                if key not in batch:
                    batch[key] = dict()
                self._concat(batch[key], frame_dict[key])
            elif isinstance(frame_dict[key], list):
                if key not in batch:
                    batch[key] = [None for _ in range(len(frame_dict[key]))]
                for i in range(len(frame_dict[key])):
                    dtype = frame_dict[key][i].dtype
                    value = frame_dict[key][i].float().mean(dim=0, keepdim=True).to(dtype)
                    batch[key][i] = torch.cat((batch[key][i], value), dim=1) if batch[key][i] is not None else value
            elif torch.is_tensor(frame_dict[key]):
                dtype = frame_dict[key].dtype
                value = frame_dict[key].float().mean(dim=0, keepdim=True).to(dtype)
                batch[key] = torch.cat((batch[key], value), dim=1) if key in batch else value


class ResultsWriter(object):
    """Since the output sub-sequences maybe overlap,
    this class cache the history animation results.
    It will render and output to video at once the history
    will be not overlapped again.
    """

    def __init__(
        self,
        pl_module,
        out_prefix: str,
        grid_size: int,
        fps: float,
        src_audio_path: str,
        draw_fn_list: Tuple[Callable, ...],
        draw_name_postfix: bool = True,
        extra_tag: Optional[str] = None,
    ):

        # * Set PL Module and it's hparams
        self.pl_module = pl_module
        self.seq_info = self.pl_module.seq_info
        self.hparams: DictConfig = pl_module.hparams  # type: ignore

        # * Set arguments
        self.out_prefix = out_prefix
        self.grid_size = grid_size
        self.fps = fps
        self.src_apath = src_audio_path
        self.draw_name_postfix = draw_name_postfix
        self.extra_tag = extra_tag

        # * Reset the states
        self.reset()

        # * Create writers
        self.create_writers(draw_fn_list)

    def reset(self):
        self.nothing_written = False
        self.curr_iframe: Optional[int] = None
        # store the results
        self.towrite: List[_ToWrite] = []
        # store metrics and variable ranges
        self.v_ranges: Dict[str, Any] = dict()
        self.metrics = dict()
        # to dump verts and coeffs
        self.to_dump_offsets: Dict[str, List[np.ndarray]] = dict()
        self.to_dump_coeffs: Dict[str, List[np.ndarray]] = dict()
        # clear writers
        if hasattr(self, "writers"):
            for x in self.writers:
                x.release()
        self.writers: List[VideoWriter] = []
        self.draw_fn_list = []
        self.draw_fn_names = []

    def create_writers(self, draw_fn_list):
        all_exists = True
        kwargs = dict(audio_source=self.src_apath, fps=float(self.fps), makedirs=True, quality="high")
        if not self.draw_name_postfix:
            assert len(draw_fn_list) == 1
        # iter fn list, create the ones we need
        for draw_fn in draw_fn_list:
            name = draw_fn.__name__
            out_prefix = self.out_prefix
            if self.extra_tag is not None and len(self.extra_tag) > 0:
                out_prefix = os.path.join(out_prefix, self.extra_tag)
                # out_prefix += f"-{self.extra_tag}"
            vpath = f"{out_prefix}({name}).mp4" if self.draw_name_postfix else f"{out_prefix}.mp4"
            if os.path.exists(vpath) and not self.overwrite_videos:
                continue
            # create writer if not exists
            self.draw_fn_list.append(draw_fn)
            self.draw_fn_names.append(name)
            self.writers.append(VideoWriter(vpath, **kwargs))
            all_exists = False
        if all_exists:
            log.warning(f"{os.path.basename(self.out_prefix)} exists.")

    def is_empty(self):
        return len(self.writers) == 0

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                            properties from hparams                                           * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    @property
    def dump_metrics(self):
        return self.hparams.visualizer.get("dump_metrics", True)

    @property
    def dump_images(self):
        return self.hparams.visualizer.get("dump_images", False)

    @property
    def dump_offsets(self):
        return self.hparams.visualizer.get("dump_offsets", False)

    @property
    def dump_coeffs(self):
        return self.hparams.visualizer.get("dump_coeffs", False)

    @property
    def overwrite_videos(self):
        return self.hparams.visualizer.get("overwrite_videos", False)

    # * ------------------------------------------------------------------------------------------------------------ * #
    # *                                       API for append and write results                                       * #
    # * ------------------------------------------------------------------------------------------------------------ * #

    def append(self, frame_ids, batch, results, hiddens):
        """Append new data, whose frame_ids is given.
        All data in should be (N,), (N,C), (N,L,...) shape.
        In (N,L,...) case, the frame dim (2n) should be 1 or same length with frame_ids.
        If some frame_id appears before, it will be cached and averaged.
        Handled by _ToWrite class.
        """

        if len(frame_ids) == 0:
            return

        # * Check contiguous
        for i in range(1, len(frame_ids)):
            assert (
                frame_ids[i - 1] + 1 == frame_ids[i]
            ), f"Given 'frame_ids' is not contiguously increasing! {frame_ids}"

        # * Find the place to put new frame_ids in `towrite_frameid`
        idx = 0
        while idx < len(self.towrite) and self.towrite[idx].frame_id != frame_ids[0]:
            idx += 1

        # * Mark all history as not updated first
        for x in self.towrite:
            x.updated = False
        # * Update history with new coming frame_ids
        for i, fid in enumerate(frame_ids):
            # * If this frame_id is new.
            if idx >= len(self.towrite):
                self.towrite.append(_ToWrite(fid))
            # * Check increasing contiguously
            assert idx == 0 or self.towrite[idx - 1].frame_id + 1 == self.towrite[idx].frame_id
            # * Update history and merge history with new coming data
            assert self.towrite[idx].frame_id == fid
            self.towrite[idx].update(
                i,
                expected_frames=len(frame_ids),
                possible_padding=self.seq_info._padding,
                batch=batch,
                results=results,
                hiddens=hiddens,
            )
            # * Increase index
            idx += 1

        # * Set first curr_iframe
        if self.curr_iframe is None:
            self.curr_iframe = frame_ids[0]

    def write_history(self, force_all=False):
        # * Get batch and results
        count, batch, results, hiddens = 0, dict(), dict(), dict()
        for i in range(len(self.towrite)):
            if self.towrite[i].updated and not force_all:
                break
            # * only get history that won't be updated again
            self.towrite[i].concat_to("batch", batch)
            self.towrite[i].concat_to("results", results)
            self.towrite[i].concat_to("hiddens", hiddens)
            count += 1
        # * Got n_frames history
        n_frames = count
        # * Return if nothing to write
        if n_frames == 0:
            return 0
        # * Pop out the got history
        while count > 0:
            assert (not self.towrite[0].updated) or force_all
            self.towrite.pop(0)
            count -= 1

        # * Neural rendering
        if hasattr(self.pl_module, "on_visualization"):
            self.pl_module.on_visualization(batch, results)

        # * Metrics
        if self.dump_metrics and hasattr(self.pl_module, "get_metrics"):
            metric_dict = self.pl_module.get_metrics(batch, results, prefix=None, reduction="none")
            for key in metric_dict:
                assert metric_dict[key].shape[0] == 1, f"invalid shape for {key}, {metric_dict[key].shape}"
                assert metric_dict[key].ndim >= 2, f"invalid shape for {key}, {metric_dict[key].shape}"
                if key not in self.metrics:
                    self.metrics[key] = list()
                self.metrics[key].extend(float(x.mean()) for x in metric_dict[key][0])

        # * Iter valid frames
        for fi in range(n_frames):
            # * Fetch final offsets and nr_fake
            to_dump_images = self.pl_module.to_dump_images(results, fi)
            to_dump_offsets = self.pl_module.to_dump_offsets(results, fi)

            cnt_written = 0
            for fn_name, draw_fn, writer in zip(self.draw_fn_names, self.draw_fn_list, self.writers):
                if writer is None:
                    continue
                # draw and write
                canvas = draw_fn(self.hparams, batch, results, 0, fi, self.grid_size)
                if canvas is None:
                    continue
                canvas_bgr = canvas[..., [2, 1, 0]]
                writer.write(canvas_bgr)
                cnt_written += 1
                # (optional) debug
                if self.hparams.debug:
                    # imutils.imshow(fn_name, canvas)
                    imutils.imwrite(fn_name + ".png", canvas)
            if cnt_written == 0:
                self.nothing_written = True
                break

            # dump image
            if self.dump_images and to_dump_images is not None:
                for key, img in to_dump_images.items():
                    # Prepare dirs
                    out_dir_img = os.path.join(self.out_prefix, key)
                    if not os.path.exists(out_dir_img):
                        os.makedirs(out_dir_img)
                    save_path = os.path.join(out_dir_img, f"{self.curr_iframe:06d}.png")
                    if torch.is_tensor(img):
                        img = img.detach().cpu().numpy()
                    img = (img + 1) / 2
                    imutils.imwrite(save_path, img)

            # dump verts delta
            if self.dump_offsets and to_dump_offsets is not None:
                for key, offsets in to_dump_offsets.items():
                    if torch.is_tensor(offsets):
                        offsets = offsets.detach().cpu().numpy()
                    # append
                    if key not in self.to_dump_offsets:
                        self.to_dump_offsets[key] = []
                    assert len(self.to_dump_offsets[key]) == self.curr_iframe
                    self.to_dump_offsets[key].append(offsets)

            # dump coeffs
            if self.dump_coeffs and results.get("coeffs_dict") is not None:
                for key in results["coeffs_dict"]:
                    coeffs = results["coeffs_dict"][key][0, fi].detach().cpu().numpy()
                    if key not in self.to_dump_coeffs:
                        self.to_dump_coeffs[key] = []
                    assert len(self.to_dump_coeffs[key]) == self.curr_iframe
                    self.to_dump_coeffs[key].append(coeffs)

            # (optional) debug
            if self.hparams.debug:
                imutils.waitKey(1)

            # increase curr_iframe
            assert self.curr_iframe is not None
            self.curr_iframe += 1

        # * Cache idle_verts for saving at calling close()
        self.idle_verts = batch["idle_verts"][0].detach().cpu().numpy()

        return n_frames

    def close(self):
        # * release writers
        for writer in self.writers:
            if writer is not None:
                writer.release()

        # Guard previous dumped results if nothing is written
        if self.nothing_written:
            log.info(f"Results saved in: {self.out_prefix}")
            return

        # * save metrics
        if self.dump_metrics and len(self.metrics) > 0:
            metrics_path = self.out_prefix + "_metrics.json"
            with open(metrics_path, "w") as fp:
                json.dump(self.metrics, fp)

        # * dump verts
        if self.dump_offsets:
            # * save idle verts
            idle_fname = self.hparams.visualizer.get("dump_idle_fname", "idle.obj")
            idle_save_path = os.path.join(self.out_prefix, idle_fname)
            os.makedirs(os.path.dirname(idle_save_path), exist_ok=True)
            save_obj(idle_save_path, self.idle_verts, get_vocaset_template_triangles())
            # * save offsets
            os.makedirs(self.out_prefix, exist_ok=True)
            # save_path = os.path.join(self.out_prefix, "dump-offsets.npz")
            # save_dict = dict()
            for key in self.to_dump_offsets:
                save_path = os.path.join(self.out_prefix, f"dump-offsets-{key}.npy")
                arr = np.asarray(self.to_dump_offsets[key], dtype=np.float32)
                # HACK: reference sequence
                if key.startswith("refer"):
                    assert arr.ndim == 4 and arr.shape[0] == 1
                    arr = arr[0]
                np.save(save_path, arr)
                log.info(f"Dump offsets: {os.path.basename(save_path)} {arr.shape}")
                # save_dict[key] = arr
            # np.savez_compressed(save_path, **save_dict)
            # np.savez(save_path, **save_dict)

        # * dump coeffs
        if self.dump_coeffs:
            os.makedirs(self.out_prefix, exist_ok=True)
            for key in self.to_dump_coeffs:
                arr = np.asarray(self.to_dump_coeffs[key], dtype=np.float32)
                np.save(os.path.join(self.out_prefix, f"dump-coeffs-{key}.npy"), arr)

        log.info(f"Results saved in: {self.out_prefix}")
