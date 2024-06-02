import json
import os

import cv2
import librosa
import numpy as np
import pandas
import soundfile as sf
import toml
from tqdm import tqdm, trange

from src.engine.logging import get_logger
from src.engine.misc import filesys
from src.engine.misc.csv import DataDict, write_csv
from src.mm_fitting.tool.face_parsing.get_masks import evaluate_video, load_network

from ..utils_audio import export_audio_features, normalize_audio

log = get_logger("TALK_VIDEO")
_face_parsing_net = None


def _tag(spk, seq):
    return f"{spk}/{seq}"


def _get_fps(vpath):
    reader = cv2.VideoCapture(vpath)
    fps = reader.get(cv2.CAP_PROP_FPS)
    reader.release()
    return fps


def _seq_id(vpath):
    return os.path.splitext(os.path.basename(vpath))[0]


def _save_into_csv(dataset_root, name, data_dicts):
    if len(data_dicts) > 0:
        write_csv(os.path.join(dataset_root, name + ".csv"), data_dicts[0].metadata, data_dicts)


def _read_seqs_info(csv_path):
    df = pandas.read_csv(csv_path, sep=",")
    # check metadata
    metadata = df.columns.values  # type: ignore
    # read tuples
    data_dict = dict()
    for row in df.values:  # type: ignore
        data = {meta: str(d) for d, meta in zip(row, metadata)}
        assert data["seq"] not in data_dict
        data_dict[data["seq"]] = data
    return data_dict


def _process_video(
    config,
    speaker,
    seq_id,
    out_dir,
    vtype,
    vpath,
    clip_source,
    avoffset_ms,
    A=256,
    wanna_fps=None,
    wanna_sr=16000,
    overwrite=False,
    tqdm_progress=None,
):

    lpath = os.path.splitext(vpath)[0] + "-lmks-fw75.toml"
    assert os.path.exists(vpath), "Failed to find {}".format(vpath)
    assert os.path.exists(lpath), "Failed to find {}".format(lpath)
    with open(lpath) as fp:
        lmks_data = toml.load(fp)
        data_fps = float(lmks_data["fps"])

    # # * Check fps
    # # fps integer times of wanna_fps
    # fps_times = data_fps / wanna_fps
    # assert fps_times.is_integer(), f"Source fps {data_fps} is not integer times of {wanna_fps}"
    # fps_times = int(fps_times)
    assert wanna_fps is None
    if speaker in config.data.CELEBTALK.VIDEO_SPEAKERS:
        assert data_fps == config.data.CELEBTALK.DATA_FPS[speaker]
    fps_times = 1

    # * Save info
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "info.json"), "w") as fp:
        # json.dump(dict(fps=wanna_fps), fp)
        json.dump(dict(fps=data_fps), fp)

    # locks dir
    locks_dir = os.path.join(out_dir, "locks")
    os.makedirs(locks_dir, exist_ok=True)

    # * Load landmarks
    if tqdm_progress is not None:
        tqdm_progress.set_description(f"[{_tag(speaker, seq_id)}] lmk75")
    save_path_lmks_fw75 = os.path.join(out_dir, "lmks_fw75.npy")
    if (not os.path.exists(save_path_lmks_fw75)) or overwrite:
        lmks_list = []
        W, H = lmks_data["resolution"]
        # ! adjust FPS
        for i_frame in range(0, len(lmks_data["frames"]), fps_times):
            points = lmks_data["frames"][i_frame]["points"]
            points = np.asarray(points, dtype=np.float32)  # type: ignore
            lmks_list.append(points)
        lmks_npy = np.asarray(lmks_list, dtype=np.float32)  # type: ignore
        lmks_npy[..., 0] = (lmks_npy[..., 0] / W) * 2.0 - 1.0  # type: ignore
        lmks_npy[..., 1] = (lmks_npy[..., 1] / H) * 2.0 - 1.0  # type: ignore
        np.save(save_path_lmks_fw75, lmks_npy)

    # * Dump images
    if tqdm_progress is not None:
        tqdm_progress.set_description(f"[{_tag(speaker, seq_id)}] image")
    image_dir = os.path.join(out_dir, "images")
    img_done_flag = os.path.join(locks_dir, "done_images.lock")
    if (not os.path.exists(img_done_flag)) or overwrite:
        image_paths = []
        reader = cv2.VideoCapture(vpath)
        assert reader.get(cv2.CAP_PROP_FPS) == data_fps
        os.makedirs(image_dir, exist_ok=True)
        frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
        for iframe in trange(frame_count, desc=f"[{_tag(speaker, seq_id)}] video", leave=False):
            ret, frame = reader.read()
            if not ret:
                break
            # ! adjust FPS
            if iframe % fps_times != 0:
                continue
            frame = cv2.resize(frame, (A, A), interpolation=cv2.INTER_LANCZOS4)
            # save
            save_path = os.path.join(image_dir, "{:d}.png".format(len(image_paths)))
            save_path = os.path.relpath(save_path)
            cv2.imwrite(save_path, frame)
            image_paths.append(save_path)
        reader.release()
        # the flag file of done
        with open(img_done_flag, "w") as fp:
            fp.write("")
    else:
        image_paths = filesys.find_files(image_dir, r"^\d+\.png$", False, False)
    n_frames = len(image_paths)

    # * Mask
    if tqdm_progress is not None:
        tqdm_progress.set_description(f"[{_tag(speaker, seq_id)}] masks")
    mask_done_flag = os.path.join(locks_dir, "done_masks.lock")
    if (not os.path.exists(mask_done_flag)) or overwrite:
        global _face_parsing_net
        if _face_parsing_net is None:
            _face_parsing_net = load_network()
        evaluate_video(_face_parsing_net, out_dir)
        # done flag
        with open(mask_done_flag, "w") as fp:
            fp.write("")

    # * Audio and features
    if tqdm_progress is not None:
        tqdm_progress.set_description(f"[{_tag(speaker, seq_id)}] audio")
    apath = os.path.join(out_dir, "audio.wav")
    if (not os.path.exists(apath)) or overwrite:
        y, sr = librosa.core.load(vpath, sr=wanna_sr)
        assert sr == wanna_sr
        # print(max(np.abs(y)))
        y = normalize_audio(y, target_db=-23, threshold=-40)
        # print(max(np.abs(y)))
        sf.write(apath, y, sr)
    else:
        sr = wanna_sr
    # Audio feature
    af_dir = os.path.join(out_dir, "audio_features")
    af_dict = export_audio_features(config.data, apath, af_dir, makedirs=True, overwrite=overwrite)

    # * Data dict
    data_dict = DataDict(
        {
            "data_dir:path": out_dir,
            # source
            "type:str": vtype,
            "tag:str": f"{speaker}/{seq_id}",
            "clip_source:str": clip_source,
            "speaker:str": speaker,
            "seq_id:str": seq_id,
            # video
            "fps:float": data_fps,  # wanna_fps
            "n_frames:int": n_frames,
            # audio
            "sample_rate:int": sr,
            "mel_hop:float": af_dict["mel_hop"],
            "ds_hop:float": af_dict["ds_hop"],
            # avoffset
            "avoffset_ms:float": avoffset_ms,
            "avoffset:int": int(np.round(avoffset_ms * data_fps / 1000.0)),  # wanna_fps
        }
    )
    with open(os.path.join(out_dir, "info.json"), "w") as fp:
        json.dump(data_dict.to_dict(), fp, indent=2)
    return data_dict, fps_times


def prepare_celebtalk(config, source_root, dataset_root):
    source_root = os.path.expanduser(source_root)
    dataset_root = os.path.expanduser(dataset_root)
    log.info("Prepare data from https://github.com/chaiyujin/CelebTalk.git")
    log.info("Source  Root: {}".format(source_root))
    log.info("Dataset Root: {}".format(dataset_root))

    if not os.path.isdir(source_root):
        log.warning("Failed to find root: '{}'".format(source_root))
        return

    # prepare dicts
    dataset_dicts = dict(train=[], valid=[], test=[])

    # * find all speakers in processed file
    assert os.path.exists(os.path.join(source_root, "Processed"))
    spk_roots = filesys.find_dirs(os.path.join(source_root, "Processed"), r"(m|f).\d\d_.+", recursive=False)
    # if specific speakers are required
    if len(config.dataset.speakers) > 0:
        spk_roots = [spk_root for spk_root in spk_roots if os.path.basename(spk_root) in config.dataset.speakers]

    spk_roots = [spk_root for spk_root in spk_roots if os.path.basename(spk_root) == "fx00_test"]

    # if some speakers' data are not ready
    spk_roots = [x for x in spk_roots if os.path.exists(os.path.join(x, "clips_cropped"))]
    print("Find {} speakers".format(len(spk_roots)))
    for x in spk_roots:
        print("-", os.path.basename(x))

    # * for each speaker
    for spk_root in spk_roots:
        spk_id = os.path.basename(spk_root)
        # speaker name
        spk_src_root = os.path.join(source_root, "Processed", spk_id)
        spk_dst_root = os.path.join(dataset_root, "data", spk_id)
        # find the sequence information from untracked root
        seqs_info_dict = _read_seqs_info(os.path.join(spk_src_root, "clips", "info.csv"))
        os.makedirs(spk_dst_root, exist_ok=True)
        # get avoffset suggest
        avoffset_dict = dict()
        with open(os.path.join(source_root, "ProcessTasks", spk_id, "avoffset_suggest.txt")) as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0 or line[0] == "#":
                    continue
                ss = line.split(",")
                assert len(ss) == 2
                key = ss[0]
                val, unit = ss[1].split()
                assert unit.lower() in ["frame", "frames"]
                avoffset_dict[key] = int(val)

        # * iter each sequence of speaker
        spk_vid_dir = os.path.join(spk_src_root, "clips_cropped")
        vpath_list = filesys.find_files(spk_vid_dir, r"^(trn|vld|tst)-\d+\.mp4$", recursive=False, abspath=True)

        vpath_list = filesys.find_files(spk_vid_dir, r"^tst-\d+\.mp4$", recursive=False, abspath=True)

        # ignores
        ignores = config.dataset.get("ignores", [])
        vpath_list = [x for x in vpath_list if _tag(spk_id, _seq_id(x)) not in ignores]
        progress = tqdm(vpath_list)
        for _, vpath in enumerate(progress):
            # seq id
            seq_id = _seq_id(vpath)
            assert seq_id[:3] in ["trn", "vld", "tst"]
            # set the tag
            tag = _tag(spk_id, seq_id)
            progress.set_description(f"[{_tag(spk_id, seq_id)}]")
            # clip source
            clip_source = seqs_info_dict[seq_id]["source"]
            assert clip_source in avoffset_dict, f"Failed to find avoffset suggestion for {spk_id}'s {clip_source}"
            assert (
                f"{spk_id}/{clip_source}" in config.data.CELEBTALK.SPEAKER_SOURCES[spk_id]
            ), f"clip_source '{clip_source}' is not in config for speaker '{spk_id}'"
            # avoffset
            fps = _get_fps(vpath)
            avoffset_ms = avoffset_dict[clip_source] * 1000.0 / fps
            # output dir
            out_dir = os.path.join(dataset_root, "data", spk_id, seq_id)

            # process this video
            data_dict, fps_times = _process_video(
                config,
                speaker=spk_id,
                seq_id=seq_id,
                out_dir=out_dir,
                vtype="celebtalk",
                vpath=vpath,
                clip_source=f"{spk_id}/{clip_source}",
                avoffset_ms=avoffset_ms,
                A=config.data.video.image_size,
                wanna_fps=None,
                wanna_sr=config.data.audio.sample_rate,
                tqdm_progress=progress,
            )
            if data_dict is None:
                continue

            # append new data_dict
            if seq_id.startswith("trn"):
                dataset_dicts["train"].append(data_dict)
            elif seq_id.startswith("vld"):
                dataset_dicts["valid"].append(data_dict)
            elif seq_id.startswith("tst"):
                dataset_dicts["test"].append(data_dict)
            else:
                raise ValueError(f"Invalid tag: {tag}")

            if config.debug:
                break
        if config.debug:
            break

    # fmt: off
    # _save_into_csv(dataset_root, "train", dataset_dicts["train"])
    # _save_into_csv(dataset_root, "valid", dataset_dicts["valid"])
    # _save_into_csv(dataset_root, "test",  dataset_dicts["test"])
    # _save_into_csv(dataset_root, "all",   dataset_dicts["train"] + dataset_dicts["valid"] + dataset_dicts["test"])
    # fmt: on


def prepare_facetalk(config, source_root, dataset_root):
    source_root = os.path.expanduser(source_root)
    dataset_root = os.path.expanduser(dataset_root)
    log.info("Prepare data from FaceTalk (VOCASET videos)")
    log.info("Source  Root: {}".format(source_root))
    log.info("Dataset Root: {}".format(dataset_root))

    if not os.path.isdir(source_root):
        log.warning("Failed to find root: '{}'".format(source_root))
        return
    if os.path.exists(os.path.join(source_root, "Data")) and os.path.exists(os.path.join(source_root, ".gitignore")):
        source_root = os.path.join(source_root, "Data")

    # prepare dicts
    dataset_dicts = dict(train=[], valid=[], test=[])

    # * find all speakers in processed file
    assert os.path.exists(os.path.join(source_root, "videos_lmks_crop"))
    spk_roots = filesys.find_dirs(os.path.join(source_root, "videos_lmks_crop"), r"^FaceTalk.*", recursive=False)
    if len(config.dataset.speakers) > 0:
        spk_roots = [spk_root for spk_root in spk_roots if os.path.basename(spk_root) in config.dataset.speakers]
    print("Find {} speakers".format(len(spk_roots)))

    # * for each speaker
    for spk_root in spk_roots:
        spk_id = os.path.basename(spk_root)
        # speaker name
        spk_src_root = os.path.join(source_root, "videos_lmks_crop", spk_id)
        spk_dst_root = os.path.join(dataset_root, "data", spk_id)
        os.makedirs(spk_dst_root, exist_ok=True)

        # * iter each sequence of speaker
        vpath_list = filesys.find_files(spk_src_root, r"sentence\d\d.mp4$", recursive=False, abspath=True)
        # ignores
        ignores = config.dataset.get("ignores", [])
        vpath_list = [x for x in vpath_list if _tag(spk_id, _seq_id(x)) not in ignores]
        progress = tqdm(vpath_list)
        for _, vpath in enumerate(progress):
            # seq id
            seq_id = _seq_id(vpath)
            assert seq_id.startswith("sentence")
            seq_number = int(seq_id[8:])
            # set the tag
            tag = _tag(spk_id, seq_id)
            progress.set_description(f"[{_tag(spk_id, seq_id)}]")
            # clip source
            clip_source = spk_id
            # avoffset
            avoffset_ms = 100.0
            # output dir
            out_dir = os.path.join(dataset_root, "data", spk_id, seq_id)

            # process this video
            data_dict, fps_times = _process_video(
                config,
                speaker=spk_id,
                seq_id=seq_id,
                out_dir=out_dir,
                vtype="facetalk",
                vpath=vpath,
                clip_source=clip_source,
                avoffset_ms=avoffset_ms,
                A=config.data.video.image_size,
                wanna_fps=None,
                wanna_sr=config.data.audio.sample_rate,
                tqdm_progress=progress,
            )
            if data_dict is None:
                continue

            # append new data_dict
            if 6 <= seq_number <= 30:
                dataset_dicts["train"].append(data_dict)
            else:
                dataset_dicts["valid"].append(data_dict)

            if config.debug:
                break
        if config.debug:
            break

    # fmt: off
    _save_into_csv(dataset_root, "train", dataset_dicts["train"])
    _save_into_csv(dataset_root, "valid", dataset_dicts["valid"])
    _save_into_csv(dataset_root, "test",  dataset_dicts["test"])
    _save_into_csv(dataset_root, "all",   dataset_dicts["train"] + dataset_dicts["valid"] + dataset_dicts["test"])
    # fmt: on
