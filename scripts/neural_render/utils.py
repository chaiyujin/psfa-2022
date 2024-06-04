import json
import os
import pickle
import re
from contextlib import contextmanager
from glob import glob
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import cv2
import lmdb
import numpy as np
import torch
import torch.cuda
import torch.nn as nn
from omegaconf import OmegaConf
from scipy.io import loadmat
from torch import Tensor
from tqdm import tqdm, trange

T_Tensor = Union[np.ndarray, Tensor]
FID_REGEX = re.compile(r"(frame)?(\d+)")


def to_tensor(x: T_Tensor, device=None, dtype=None) -> Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x.to(device=device, dtype=dtype)


def to_device(data: Dict[str, Tensor], device: str) -> Dict[str, Tensor]:
    for k in data:
        data[k] = to_tensor(data[k], device=device)
    return data


def fid_of(filepath: str):
    term = os.path.splitext(os.path.basename(filepath))[0]
    match = FID_REGEX.match(term)
    assert match is not None
    return int(match.group(2))


class _RendererLoader:
    g_instance: Optional[nn.Module] = None
    g_load_from: str = ""

    @classmethod
    def load_renderer(cls, name, load_from, device):
        # clear cached
        cls.release()

        if name == "ours":
            from src.modules.neural_renderer import NeuralRenderer

            # find hparams
            exp_dir = os.path.dirname(os.path.dirname(load_from))
            hparams_path = os.path.join(exp_dir, "tb", "hparams.yaml")
            assert os.path.exists(hparams_path), "failed to find {}".format(hparams_path)
            config = OmegaConf.load(hparams_path)
            # build model
            model = NeuralRenderer(config.model.neural_renderer)
            # load state_dict
            load_dict = torch.load(load_from, map_location="cpu")
            state_dict = dict()
            for key, val in load_dict["state_dict"].items():
                if key.startswith("renderer."):
                    new_key = key[len("renderer.") :]
                    state_dict[new_key] = val
            model.load_state_dict(state_dict)
            # set to global instance
            cls.g_instance = model.to(device)
            cls.g_instance.eval()
        elif name == "nvpp":
            from . import nvpp

            model = nvpp.NeuralRenderer(load_from)
            # set to global instance
            cls.g_instance = model
            cls.g_instance.eval()
        elif name == "stya":
            from . import stya

            model = stya.Renderer()
            model.load_pretrained(load_from)
            # set to global instance
            cls.g_instance = model.to(device)
            cls.g_instance.eval()
        else:
            # TODO: build the renderer
            raise NotImplementedError("Unknown renderer for: {}".format(name))

        # set load_from path
        cls.g_load_from = load_from

    @classmethod
    def release(cls):
        if cls.g_instance is not None:
            del cls.g_instance
            cls.g_instance = None
            cls.g_load_from = ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def maybe_reload(cls, name, load_from, device):
        need_reloading = (cls.g_instance is None) or (cls.g_load_from != load_from)
        if need_reloading:
            cls.load_renderer(name, load_from, device)


@contextmanager
def load_renderer(name: str, load_from: str, device: str, cache: bool):
    try:
        _RendererLoader.maybe_reload(name, load_from, device)
        yield _RendererLoader.g_instance
    finally:
        if not cache:
            _RendererLoader.release()


class _VideoLoader:
    g_instance: Optional[List[np.ndarray]] = None
    g_load_from: str = ""

    @classmethod
    def load(cls, _, load_from, device):
        # clear cached data
        cls.release()

        if os.path.isdir(load_from):
            files = glob("*.png", root_dir=load_from)
            files = sorted(files, key=lambda x: int(os.path.basename(x[:-4])))
            im_list = []
            for f in files:
                f = os.path.join(load_from, f)
                frame = cv2.imread(f)
                im = cv2.resize(frame, (256, 256))
                im = im[..., [2, 1, 0]].astype(np.float32) * 2.0 / 255.0 - 1.0
                im_list.append(im.transpose(2, 0, 1))  # rgb, -1~1, CHW
        else:
            # load video frames
            assert load_from.find("avoffset_corrected") >= 0
            assert os.path.exists(load_from), "Failed to find video from: '{}'".format(load_from)
            reader = cv2.VideoCapture(load_from)
            n_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
            im_list = []
            for _ in trange(n_frames, desc="Read Video Frames", leave=False):
                got, frame = reader.read()
                if not got:
                    break
                im = cv2.resize(frame, (256, 256))
                im = im[..., [2, 1, 0]].astype(np.float32) * 2.0 / 255.0 - 1.0
                im_list.append(im.transpose(2, 0, 1))  # rgb, -1~1, CHW
            reader.release()

        # set to global instance
        cls.g_instance = im_list
        cls.g_load_from = load_from

    @classmethod
    def release(cls):
        if cls.g_instance is not None:
            del cls.g_instance
            cls.g_instance = None
            cls.g_load_from = ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def maybe_reload(cls, name, load_from, device):
        need_reloading = (cls.g_instance is None) or (cls.g_load_from != load_from)
        if need_reloading:
            cls.load(name, load_from, device)


class _CoeffLoader:
    g_instance: Optional[Dict[str, Any]] = None
    g_load_from: str = ""

    @classmethod
    def load(cls, name, load_from, device):
        # clear cached
        cls.release()

        if name == "ours":
            coe_dir = os.path.join(load_from, "frames")
            assert os.path.exists(coe_dir)
            dat_dir = load_from.replace("fitted/", "")
            info_json = os.path.join(dat_dir, "info.json")
            assert os.path.exists(info_json)
            # load info
            with open(os.path.join(dat_dir, "info.json")) as fp:
                info = json.load(fp)
            # load all coeffs files
            coe_files = sorted(glob(os.path.join(coe_dir, "*.npz")), key=fid_of)
            rot_list, tsl_list, cam_list = [], [], []
            for coe_fpath in tqdm(coe_files, desc="Load Fitted Coeffs", leave=False):
                coe_dict = np.load(coe_fpath)
                rot_list.append(coe_dict["rot"])  # type: ignore
                tsl_list.append(coe_dict["tsl"])  # type: ignore
                cam_list.append(coe_dict["cam"])  # type: ignore

            # > correct avoffset
            def _correct_avoffset(x):
                avoffset = info["avoffset"]
                if avoffset > 0:  # delay visual things
                    padding = x[:1].repeat(avoffset, axis=0)
                    x = np.concatenate((padding, x), axis=0)
                elif avoffset < 0:
                    x = x[-avoffset:]
                return x

            rot_list = _correct_avoffset(np.asarray(rot_list))  # type: ignore
            tsl_list = _correct_avoffset(np.asarray(tsl_list))  # type: ignore
            cam_list = _correct_avoffset(np.asarray(cam_list))  # type: ignore

            # resample
            rot_list = interpolate_features(rot_list, info["fps"], 25)  # type: ignore
            tsl_list = interpolate_features(tsl_list, info["fps"], 25)  # type: ignore
            cam_list = interpolate_features(cam_list, info["fps"], 25)  # type: ignore
            # set to global instance
            cls.g_instance = dict(
                rot=torch.tensor(rot_list, dtype=torch.float32, device=device),
                tsl=torch.tensor(tsl_list, dtype=torch.float32, device=device),
                cam=torch.tensor(cam_list, dtype=torch.float32, device=device),
            )
        elif name == "nvpp":

            coe_dir = os.path.join(load_from, "crop")
            assert os.path.exists(coe_dir)
            _id_list, exp_list, rot_list, tsl_list, trans_list = [], [], [], [], []
            mat_files = sorted(glob(os.path.join(coe_dir, "frame*.mat")), key=fid_of)
            for i, mat_fpath in enumerate(mat_files):
                assert i == fid_of(mat_fpath)
                trans_npy = os.path.splitext(mat_fpath)[0] + "-trans.npy"
                # load coeffs
                d = loadmat(mat_fpath)
                _id_list.append(d["id"])
                exp_list.append(d["exp"])
                rot_list.append(d["angle"])
                tsl_list.append(d["trans"])
                # load transform parameters (full image -> crop image)
                trans_list.append(np.load(trans_npy, allow_pickle=True))
            # set to global instance
            cls.g_instance = dict(
                _id=np.concatenate(_id_list),
                exp=np.concatenate(exp_list),
                rot=np.concatenate(rot_list),
                tsl=np.concatenate(tsl_list),
                trans=trans_list,
            )

        elif name == "stya":
            tmp_dir = ".snaps/stya_tmp"

            def _get_video(fpath):
                cap = cv2.VideoCapture(fpath)
                frame_list = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    frame_list.append(frame)
                frame_list = np.asarray(frame_list, dtype=np.uint8)  # type: ignore
                return frame_list

            def _load_frames(data_bytes):
                os.makedirs(tmp_dir, exist_ok=True)
                with open(os.path.join(tmp_dir, "tmp.mp4"), "wb") as fp:
                    fp.write(data_bytes)
                return _get_video(os.path.join(tmp_dir, "tmp.mp4"))

            assert load_from.find(":") > 0
            lmdb_path, key, vid = load_from.split(":")  # type: ignore
            assert key in ["train", "test"]
            # parse train/test and the data_key
            # print(lmdb_path)
            # print(vid)
            _, data_key = vid.split("-")  # type: ignore
            data_key = str(int(data_key)).encode()
            # check lmdb exist
            assert os.path.exists(os.path.join(lmdb_path, "data.mdb")), "Not a valid lmdb: {}".format(lmdb_path)
            env = lmdb.open(lmdb_path, map_size=1099511627776, max_dbs=64)
            video_db = env.open_db(f"{key}_video".encode())
            coeff_db = env.open_db(f"{key}_coeff".encode())
            trans_db = env.open_db(f"{key}_trans".encode())
            with env.begin(write=False) as txn:
                video_npy = _load_frames(txn.get(data_key, db=video_db))  # type: ignore
                coeff_npy = pickle.load(BytesIO(txn.get(data_key, db=coeff_db)))  # type: ignore
                trans_npy = pickle.load(BytesIO(txn.get(data_key, db=trans_db)))  # type: ignore
            # to tensor
            video = torch.from_numpy(video_npy).float().permute(0, 3, 1, 2) / 128 - 1
            # set to global instance
            cls.g_instance = dict(
                video=video.to(device),
                coeff=coeff_npy,
                trans=trans_npy,
            )
        else:
            # TODO: load data
            raise NotImplementedError("Unknown data for: {}".format(name))

        # set load_from path
        cls.g_load_from = load_from

    @classmethod
    def release(cls):
        if cls.g_instance is not None:
            del cls.g_instance
            cls.g_instance = None
            cls.g_load_from = ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @classmethod
    def maybe_reload(cls, name, load_from, device):
        need_reloading = (cls.g_instance is None) or (cls.g_load_from != load_from)
        if need_reloading:
            cls.load(name, load_from, device)


@contextmanager
def load_data(name: str, video_path: str, coeffs_dir: str, device: str, cache: bool):
    try:
        _VideoLoader.maybe_reload(name, video_path, device)
        _CoeffLoader.maybe_reload(name, coeffs_dir, device)
        assert _VideoLoader.g_instance is not None
        assert _CoeffLoader.g_instance is not None
        data = {"img": _VideoLoader.g_instance, **_CoeffLoader.g_instance}
        n_frames = min(len(v) for _, v in data.items())
        yield {k: v[:n_frames] for k, v in data.items()}
    finally:
        if not cache:
            _CoeffLoader.release()


def fetch_batch(data: Tensor, i: int, j: int, static_frame: Optional[int]):
    assert j > i
    n_frames = len(data)
    if static_frame is not None:
        # use static frame
        k = static_frame % n_frames
        ret = to_tensor(data[k : k + 1])
        return ret.expand(j - i, *([-1] * (ret.ndim - 1)))
    else:
        # use dynamic frames, bounce at boundary
        new_list = []
        for k in range(i, j):
            # make sure the direction
            d = k // n_frames
            if d % 2 == 0:  # increasing direction
                k = k % n_frames
            else:  # decreasing direction
                k = (n_frames - 1) - (k % n_frames)
            new_list.append(to_tensor(data[k]))
        return torch.stack(new_list, dim=0)


def fetch_list(data: List[Any], i: int, j: int, static_frame: Optional[int]):
    assert j > i
    n_frames = len(data)
    if static_frame is not None:
        # use static frame
        k = static_frame % n_frames
        return [data[k] for _ in range(n_frames)]
    else:
        # use dynamic frames, bounce at boundary
        new_list = []
        for k in range(i, j):
            # make sure the direction
            d = k // n_frames
            if d % 2 == 0:  # increasing direction
                k = k % n_frames
            else:  # decreasing direction
                k = (n_frames - 1) - (k % n_frames)
            new_list.append(data[k])
        return new_list


def interpolate_features(features, input_rate, output_rate, output_len=None):
    origin_shape = list(features.shape[1:])
    origin_dtype = features.dtype
    features = np.reshape(features, (len(features), -1))
    # interpolate
    num_features = features.shape[1]
    input_len = features.shape[0]
    seq_len = input_len / float(input_rate)
    if output_len is None:
        output_len = int(seq_len * output_rate)
    input_timestamps = np.arange(input_len) / float(input_rate)
    output_timestamps = np.arange(output_len) / float(output_rate)
    output_features = np.zeros((output_len, num_features))
    for feat in range(num_features):
        output_features[:, feat] = np.interp(output_timestamps, input_timestamps, features[:, feat])
    # reshape back
    output_features = np.reshape(output_features, [len(output_features)] + origin_shape)
    return output_features.astype(origin_dtype)
