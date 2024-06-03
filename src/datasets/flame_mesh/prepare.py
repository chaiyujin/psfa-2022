import json
import logging
import os
from shutil import copyfile
from typing import Dict, List, Optional, Tuple

import cv2
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm, trange

from assets import get_vocaset_template_triangles, get_vocaset_template_vertices
from src.cpp import deformation
from src.data import image as imutils
from src.data.mesh import io as meshio
from src.data.video import VideoWriter
from src.engine.mesh_renderer import render
from src.engine.misc import filesys
from src.engine.misc.csv import DataDict, write_csv
from src.engine.painter import Text, color_mapping, put_texts

from ..utils_audio import export_audio_features, normalize_audio

logger = logging.getLogger(__name__)
deformation.set_mode("triangles")
_root = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(_root, "selection", "FLAME_noeyeballs_front_face.txt")) as fp:
    line = " ".join(x.strip() for x in fp.readlines())
    _indices = [int(x) for x in line.split()]
with open(os.path.join(_root, "selection", "AREA_EYES_ABOVE_VIDX4FULL.txt")) as fp:
    line = " ".join(x.strip() for x in fp.readlines())
    _vidx_eye = [int(x) for x in line.split()]
_indices = sorted(list(set(_indices)))
_indices_inv = [i for i in range(3931) if i not in _indices]  # + _indices_eyes
_indices_inv = sorted(list(set(_indices_inv)))


def _correct_eyeballs(key, tmpl, offsets, tmpl_ref, offsets_ref):
    assert offsets.ndim == 2
    assert offsets_ref.ndim == 2

    def _get_eyeballs_joint(inp):
        return np.stack((inp[3931:4477].mean(0), inp[4477:5023].mean(0)), axis=0)

    v_ref = offsets_ref + tmpl_ref
    v_new = offsets + tmpl

    ref_ej = _get_eyeballs_joint(v_ref)
    ref_er = v_ref[_vidx_eye]
    X = np.linalg.lstsq(ref_er.T, ref_ej.T)[0]

    J = np.matmul(v_new[_vidx_eye].T, X).T
    v_new[3931:4477] += J[-2] - v_new[3931:4477].mean(0)
    v_new[4477:5023] += J[-1] - v_new[4477:5023].mean(0)
    offsets = v_new - tmpl

    return offsets


def _save_into_csv(dataset_root, name, data_dicts):
    if len(data_dicts) > 0:
        write_csv(os.path.join(dataset_root, name + ".csv"), data_dicts[0].metadata, data_dicts)


def prepare(config, dataset_name):
    assert dataset_name in ["vocaset", "coma"]

    source_root = os.path.expanduser(config.dataset.source_root)
    dataset_root = os.path.expanduser(config.dataset.root)
    logger.info("Prepare data for FLAME")
    logger.info("Source  Root: {}".format(source_root))
    logger.info("Dataset Root: {}".format(dataset_root))

    out_root = os.path.join(dataset_root, f"{dataset_name}_data")
    dataset_dicts = dict(train=[], valid=[], test=[])

    overwrite = config.get("overwrite", False)

    pbar_spk = tqdm(config.data.VOCASET.REAL3D_SPEAKERS + config.data.VOCASET.VIDEO_SPEAKERS)
    for i_spk, spk in enumerate(pbar_spk):
        # if spk not in ['FaceTalk_170908_03277_TA']:
        #     continue

        pbar_spk.set_description(spk)
        spk_dir = os.path.join(source_root, spk)

        audio_dir = None
        if dataset_name == "vocaset":
            audio_dir = os.path.join(os.path.dirname(source_root), "audio", spk)
            assert os.path.exists(audio_dir), "Failed to find audio dir: {}".format(audio_dir)

        # * get idle
        idle_verts = get_vocaset_template_vertices(spk)
        tris = get_vocaset_template_triangles()

        def _save_idle(prefix):
            os.makedirs(os.path.dirname(prefix), exist_ok=True)
            np.save(prefix + ".npy", idle_verts)
            meshio.save_obj(prefix + ".obj", idle_verts, tris)

        _save_idle(os.path.join(out_root, spk, "identity"))

        # * iter sequence
        subdirs = filesys.find_dirs(os.path.join(spk_dir), r".*", recursive=False, abspath=True)
        if len(subdirs) == 0:
            subdirs = filesys.find_files(os.path.join(spk_dir), r".*-offsets\.npy", abspath=True)
            subdirs = [x.replace("-offsets.npy", "") for x in subdirs]

        pbar_seq = tqdm(subdirs, desc="sequence", leave=False)
        for seq_dir in pbar_seq:
            seq_id = os.path.basename(seq_dir)
            pbar_seq.set_description(seq_id)

            # dst paths
            output_dir = os.path.join(out_root, spk, seq_id)
            npy_path = os.path.join(output_dir, "offsets.npy")
            os.makedirs(output_dir, exist_ok=True)

            # * (optional) audio feature
            mel_hop, ds_hop = 0.0, 0.0
            audio_path = None
            if audio_dir is not None:
                audio_path = os.path.join(output_dir, "audio.wav")
                wanna_sr = config.data.audio.sample_rate
                if (not os.path.exists(audio_path)) or overwrite:
                    src_apath = os.path.join(audio_dir, seq_id + ".wav")
                    assert os.path.exists(src_apath), "Failed to find audio: {}".format(src_apath)
                    y, sr = librosa.core.load(src_apath, sr=wanna_sr)
                    assert sr == wanna_sr
                    # print(max(np.abs(y)))
                    y = normalize_audio(y, target_db=-23, threshold=-40)
                    # print(max(np.abs(y)))
                    sf.write(audio_path, y, sr)
                else:
                    sr = wanna_sr

                # export audio feature
                audio_feats = export_audio_features(config.data, audio_path, os.path.join(output_dir, "audio_features"))
                mel_hop = audio_feats["mel_hop"]
                ds_hop = audio_feats["ds_hop"]

            # * iter frames if not cached
            def _frame_id(fpath):
                return int(os.path.basename(os.path.splitext(fpath)[0]).replace(seq_id, "").replace(".", ""))

            def _load_from_raw_ply():
                verts_list = []
                files = filesys.find_files(seq_dir, r".*\.ply")
                for i, fpath in enumerate(tqdm(files, desc="Load .ply", leave=False)):
                    i_frame = _frame_id(fpath)
                    assert i + 1 == i_frame
                    verts = meshio.load_mesh(fpath)[0].astype(np.float32)
                    verts_list.append(verts)
                verts_list = np.asarray(verts_list, dtype=np.float32)

                # modify idle for this seq
                seq_idle_verts = np.copy(idle_verts)
                # set the non-important part
                # seq_idle_verts[_indices_inv] = verts_list[:, _indices_inv].mean(axis=0)

                return seq_idle_verts, verts_list

            def _load_from_npy():
                seq_idle_verts, _, _ = meshio.load_mesh(os.path.join(spk_dir, "idle.ply"))
                offsets = np.load(seq_dir + "-offsets.npy")
                return seq_idle_verts, offsets + seq_idle_verts[None, ...]

            def _load_source_sequence():
                if os.path.exists(seq_dir + "-offsets.npy"):
                    seq_idle_verts, verts_list = _load_from_npy()
                else:
                    seq_idle_verts, verts_list = _load_from_raw_ply()
                return seq_idle_verts, verts_list

            if not os.path.exists(npy_path):  # or hparams.get("debug", False):
                _save_idle(os.path.join(output_dir, "identity"))

                seq_idle_verts, verts_list = _load_source_sequence()
                if dataset_name == "coma":
                    cnsts_vidx = sorted(list(set(_indices_inv)))
                    full_tris = tris.astype(np.uint32)
                    deformation.set_ref_source(seq_idle_verts, full_tris)
                    deformation.set_tar_source(idle_verts, full_tris, cnsts=np.asarray(cnsts_vidx, dtype=np.uint32))

                writer = VideoWriter(
                    os.path.join(output_dir + "-video_debug.mp4"), fps=60, src_audio_path=audio_path, high_quality=True
                )
                offsets_list = []
                for verts in tqdm(verts_list, desc="Debug video", leave=False):
                    if dataset_name == "vocaset":
                        offsets_delta = (verts - seq_idle_verts).astype(np.float32)
                    elif dataset_name == "coma":
                        offsets_delta = (verts - seq_idle_verts).astype(np.float32)
                        avg = offsets_delta[_indices_inv].mean(axis=0)
                        offsets_delta -= avg[None, ...]
                        # dgrad = deformation.get_dgrad(verts)
                        # deformed = deformation.get_deformed(dgrad)
                        # offsets_delta = deformed - idle_verts
                        # offsets_delta = _correct_eyeballs("", idle_verts, offsets_delta, seq_idle_verts, verts - seq_idle_verts)
                    else:
                        raise NotImplementedError()
                    offsets_list.append(offsets_delta)

                    # debug
                    img = render(offsets_delta + idle_verts, A=512)
                    img = (img * 255).astype(np.uint8)[..., [2, 1, 0]]
                    # fmt: on
                    writer.write(img)
                    if config.get("debug"):
                        imutils.imshow("img", img)
                        imutils.waitKey(1)
                writer.release()
                np.save(npy_path, np.asarray(offsets_list, dtype=np.float32))

            n_frames = len(np.load(npy_path, mmap_mode="r"))

            # * get the data_dict
            data_dict = DataDict(
                {
                    "data_dir:path": output_dir,
                    # source
                    "type:str": dataset_name,
                    "tag:str": f"{spk}/{seq_id}",
                    "clip_source:str": spk,
                    "speaker:str": spk,
                    "seq_id:str": seq_id,
                    # video
                    "fps:float": 60,
                    "n_frames:int": n_frames,
                    # audio
                    "mel_hop:float": mel_hop,
                    "ds_hop:float": ds_hop,
                    # ! Manually measured avoffset
                    "avoffset_ms:float": 100.0,
                    "avoffset:int": 6,
                }
            )
            with open(os.path.join(output_dir, "info.json"), "w") as fp:
                json.dump(data_dict.to_dict(), fp, indent=2)

            # * split into datasets
            if spk in config.data.VOCASET.REAL3D_SPEAKERS:
                dataset_dicts["train"].append(data_dict)
            elif spk in config.data.VOCASET.VIDEO_SPEAKERS:
                dataset_dicts["valid"].append(data_dict)
            else:
                raise NotImplementedError()
            # end for seq
            if config.get("debug"):
                break
        # end for spk
        if config.get("debug"):
            break

    # fmt: off
    _save_into_csv(dataset_root, f"{dataset_name}_train", dataset_dicts["train"])
    _save_into_csv(dataset_root, f"{dataset_name}_valid", dataset_dicts["valid"])
    _save_into_csv(dataset_root, f"{dataset_name}_test",  dataset_dicts["test"])
    _save_into_csv(dataset_root, f"{dataset_name}_all",   dataset_dicts["train"] + dataset_dicts["valid"] + dataset_dicts["test"])
    # fmt: on


def _dtw(src, tar):
    scores = np.full((src.shape[0] + 1, tar.shape[0] + 1), float("inf"))
    scores[0, 0] = 0

    for i in range(1, src.shape[0] + 1):
        for j in range(1, tar.shape[0] + 1):
            cost = np.linalg.norm(src[i - 1] - tar[j - 1], axis=-1)
            # take last min from a square box
            last_min = np.min([scores[i - 1, j], scores[i, j - 1], scores[i - 1, j - 1]])
            scores[i, j] = cost + last_min

    # find the best path
    i, j = src.shape[0], tar.shape[0]
    best_path = []
    while not (i == 0 and j == 0):
        best_path.append((i - 1, j - 1))
        idx = np.argmin([scores[i - 1, j], scores[i, j - 1], scores[i - 1, j - 1]])
        if idx == 0:
            i, j = i - 1, j
        elif idx == 1:
            i, j = i, j - 1
        else:
            i, j = i - 1, j - 1
    best_path = list(reversed(best_path))
    # print(best_path)

    # find the index
    def _find_index(sync_idx):
        other_idx = 0 if sync_idx == 1 else 1
        k = 0
        index = []
        for j in range(scores.shape[sync_idx] - 1):
            assert k < len(best_path)
            ks = k
            ke = k + 1
            while ke < len(best_path) and best_path[ke][sync_idx] == j:
                ke += 1
            j_map_from = [best_path[x][other_idx] for x in range(ks, ke)]
            for idx, x in enumerate(j_map_from):
                assert x == j_map_from[0] + idx
            index.append((j_map_from[0], j_map_from[-1] + 1))
            k = ke
        return np.asarray(index, dtype=np.int32)

    map_idx_tgt_from_src = _find_index(1)
    map_idx_src_from_tgt = _find_index(0)
    return map_idx_tgt_from_src, map_idx_src_from_tgt


def prepare_dtw_align(config, dataset_name):
    assert dataset_name in ["vocaset"]

    dataset_root = os.path.expanduser(config.dataset.root)
    text_root = os.path.join(config.dataset.text_root)
    data_root = os.path.join(dataset_root, "vocaset_data")
    logger.info("DTW align data for VOCASET")
    logger.info("Data root: {}".format(data_root))

    overwrite = config.get("overwrite", False)
    speakers = sorted(list(config.data.VOCASET.REAL3D_SPEAKERS) + list(config.data.VOCASET.VIDEO_SPEAKERS))

    def _process(src_spk, i_sent, tgt_spk, j_sent):
        src_dir = os.path.join(data_root, src_spk, f"sentence{i_sent+1:02d}")
        tgt_dir = os.path.join(data_root, tgt_spk, f"sentence{j_sent+1:02d}")
        if not os.path.exists(src_dir) or not os.path.exists(tgt_dir):
            return False

        sav_pre_tfs = os.path.join(tgt_dir, "dtw_mel", f"from-{src_spk}-sentence{i_sent+1:02d}")
        sav_pre_sft = os.path.join(src_dir, "dtw_mel", f"from-{tgt_spk}-sentence{j_sent+1:02d}")
        sav_npy_tfs = sav_pre_tfs + "-idx.npy"
        sav_npy_sft = sav_pre_sft + "-idx.npy"

        def _debug_image(sav_pre, map_idx, src_mel, tgt_mel, src_spk, tgt_spk):
            aln_mel = src_mel[[sum(se) // 2 for se in map_idx]]
            img_tgt = np.clip(color_mapping(np.flip(tgt_mel, axis=1).transpose(1, 0)), 0, 1)
            img_aln = np.clip(color_mapping(np.flip(aln_mel, axis=1).transpose(1, 0)), 0, 1)
            w = img_tgt.shape[1]
            img_tgt = put_texts(img_tgt, [Text(tgt_spk, (w // 2, 0), ("center", "top"))])
            img_aln = put_texts(img_aln, [Text(src_spk + "(aligned)", (w // 2, 0), ("center", "top"))])
            img = np.concatenate((img_tgt, img_aln), axis=0) * 255.0
            cv2.imwrite(sav_pre + "-mel.png", img[..., [2, 1, 0]])

        if (not os.path.exists(sav_npy_tfs)) or (not os.path.exists(sav_npy_sft)) or overwrite:
            src_mel = np.load(os.path.join(src_dir, "audio_features", "mel.npy"))[..., 0]
            tgt_mel = np.load(os.path.join(tgt_dir, "audio_features", "mel.npy"))[..., 0]
            # align
            map_idx_tfs, map_idx_sft = _dtw(src_mel, tgt_mel)
            os.makedirs(os.path.dirname(sav_pre_tfs), exist_ok=True)
            os.makedirs(os.path.dirname(sav_pre_sft), exist_ok=True)
            np.save(sav_npy_tfs, map_idx_tfs)
            np.save(sav_npy_sft, map_idx_sft)
            # debug img
            _debug_image(sav_pre_tfs, map_idx_tfs, src_mel, tgt_mel, src_spk, tgt_spk)
            _debug_image(sav_pre_sft, map_idx_sft, tgt_mel, src_mel, tgt_spk, src_spk)

        _dump_aligned_anime(sav_pre_tfs, src_dir, tgt_dir, overwrite)
        _dump_aligned_anime(sav_pre_sft, tgt_dir, src_dir, overwrite)

        # debug video
        # _dump_aligned_video(sav_pre_tfs, src_dir, tgt_dir, overwrite)
        # _dump_aligned_video(sav_pre_sft, tgt_dir, src_dir, overwrite)

        return True

    def _load_texts(spk):
        texts = []
        with open(os.path.join(text_root, spk + ".txt")) as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0:
                    continue
                line = line.lower()
                line = line.replace("?", "").replace(".", "").replace(",", "").replace("'", "")
                texts.append(line)
        return texts

    jobs = []
    for i_src, src_spk in enumerate(speakers):
        for i_tgt, tgt_spk in enumerate(speakers):
            if i_tgt <= i_src:
                continue
            src_texts = _load_texts(src_spk)
            tgt_texts = _load_texts(tgt_spk)
            for i, s_txt in enumerate(src_texts):
                for j, t_txt in enumerate(tgt_texts):
                    if s_txt == t_txt:
                        jobs.append((src_spk, i, tgt_spk, j, s_txt))
            # continue
    all_pairs = []
    pbar = tqdm(jobs)
    for src_spk, i_sent, tgt_spk, j_sent, txt in pbar:
        desc = f"{src_spk[16:]}({i_sent+1:02d}) -> {tgt_spk[16:]}({j_sent+1:02d}), {txt}"
        pbar.set_description(desc)
        # do job
        if _process(src_spk, i_sent, tgt_spk, j_sent):
            all_pairs.append(f"{src_spk}-sentence{i_sent+1:02d},{tgt_spk}-sentence{j_sent+1:02d}")
    with open(os.path.join(data_root, "pairs.json"), "w") as fp:
        json.dump(all_pairs, fp, indent=2)


def _dump_aligned_video(sav_pre, src_dir, tgt_dir, overwrite):
    # fancy colors
    green = (186, 244, 173)

    src_spk = os.path.basename(os.path.dirname(src_dir))
    tgt_spk = os.path.basename(os.path.dirname(tgt_dir))

    idx_npy = sav_pre + "-idx.npy"
    opath = sav_pre + "-vid.mp4"
    if os.path.exists(opath) and (not overwrite):
        return

    map_idx = np.load(idx_npy)
    mel_indices = [list(range(len(map_idx))), [sum(se) // 2 for se in map_idx]]
    vpath_lists = [tgt_dir + "-video_debug.mp4", src_dir + "-video_debug.mp4"]
    tags = [tgt_spk, src_spk + " (aligned)"]
    readers = [cv2.VideoCapture(x) for x in vpath_lists]
    cached: Dict[int, Optional[Tuple[int, np.ndarray]]] = {i: None for i in range(len(readers))}
    mel_hop = 0.008

    def _read(i_reader, i_frame):
        # get the sec for this reader
        sec_sync = i_frame / fps
        imel_sync = int(np.round(sec_sync / mel_hop))
        imel_sync = np.clip(imel_sync, 0, len(mel_indices[i_reader]) - 1)
        imel = mel_indices[i_reader][imel_sync]
        sec = imel * mel_hop
        ifrm = int(np.round(sec * fps))
        # check cached
        while cached[i_reader] is None or cached[i_reader][0] < ifrm:
            ret, frame = readers[i_reader].read()
            if not ret:
                break
            idx = int(readers[i_reader].get(cv2.CAP_PROP_POS_FRAMES))
            w = frame.shape[1]
            img = np.pad(frame, [[20, 0], [0, 0], [0, 0]], "constant")
            img = put_texts(img, [Text(tags[i_reader], (w // 2, 0), ("center", "top"), green)], font_size=18)
            cached[i_reader] = (idx, img)
        if cached[i_reader] is None:
            return None
        else:
            return cached[i_reader][1]

    apath = os.path.join(tgt_dir, "audio.wav")
    fps = readers[0].get(cv2.CAP_PROP_FPS)
    n_frames = int(readers[0].get(cv2.CAP_PROP_FRAME_COUNT))
    writer = VideoWriter(opath, fps, src_audio_path=apath, high_quality=True)
    for i_frm in range(n_frames):
        # read images
        imgs = []
        for i in range(len(readers)):
            imgs.append(_read(i, i_frm))

        if any(x is None for x in imgs):
            break

        canvas = np.concatenate((imgs), axis=1)
        writer.write(canvas)
    writer.release()
    for x in readers:
        x.release()


def _dump_aligned_anime(sav_pre, src_dir, tgt_dir, overwrite):
    src_spk = os.path.basename(os.path.dirname(src_dir))
    tgt_spk = os.path.basename(os.path.dirname(tgt_dir))

    idx_npy = sav_pre + "-idx.npy"
    opath = sav_pre + "-off.npy"
    vpath = sav_pre + "-off_vid.mp4"

    # HACK: hard-coded fps and mel_hop
    fps = 60.0
    mel_hop = 0.008

    if (not os.path.exists(opath)) or overwrite:
        map_idx = np.load(idx_npy)
        tgt_off = np.load(os.path.join(tgt_dir, "offsets.npy"))
        src_off = np.load(os.path.join(src_dir, "offsets.npy"))

        mel_idx = [sum(se) // 2 for se in map_idx]
        aln_off = []
        for kfrm in range(len(tgt_off)):
            kmel = kfrm / fps / mel_hop
            kmel, alpha = int(kmel), kmel - int(kmel)
            # fmt: off
            imel = mel_idx[np.clip(kmel,     0, len(mel_idx) - 1)]
            jmel = mel_idx[np.clip(kmel + 1, 0, len(mel_idx) - 1)]
            ifrm = imel * mel_hop * fps
            jfrm = jmel * mel_hop * fps
            # fmt: on
            idx = ifrm * (1 - alpha) + jfrm * alpha
            # get the offset by interp
            i, j, a = int(idx), int(idx) + 1, idx - int(idx)
            i = np.clip(i, 0, len(src_off) - 1)
            j = np.clip(j, 0, len(src_off) - 1)
            off = src_off[i] * (1 - a) + src_off[j] * a
            aln_off.append(off)
        aln_off = np.asarray(aln_off, dtype=np.float32)
        assert len(aln_off) == len(tgt_off)

        # save aligned offsets
        os.makedirs(os.path.dirname(opath), exist_ok=True)
        np.save(opath, aln_off)

    if (not os.path.exists(vpath)) or overwrite:
        tgt_idle, tris, _ = meshio.load_mesh(os.path.join(tgt_dir, "identity.obj"))
        src_idle, tris, _ = meshio.load_mesh(os.path.join(src_dir, "identity.obj"))
        tgt_off = np.load(os.path.join(tgt_dir, "offsets.npy"))
        aln_off = np.load(opath)

        apath = os.path.join(tgt_dir, "audio.wav")
        writer = VideoWriter(vpath, fps, src_audio_path=apath, high_quality=True, makedirs=True)
        n_frames = len(tgt_off)
        for i in range(n_frames):
            im0 = render(tgt_idle + tgt_off[i], A=256)
            im1 = render(src_idle + aln_off[i], A=256)
            im0 = np.pad(im0, [[14, 0], [0, 0], [0, 0]], "constant")
            im1 = np.pad(im1, [[14, 0], [0, 0], [0, 0]], "constant")
            im0 = put_texts(im0, [Text(tgt_spk, (im0.shape[1] // 2, 0), ("center", "top"))], font_size=12)
            im1 = put_texts(
                im1, [Text(src_spk + " (aligned)", (im1.shape[1] // 2, 0), ("center", "top"))], font_size=12
            )
            canvas = np.concatenate((im0, im1), axis=1)
            writer.write(canvas[..., [2, 1, 0]])
        writer.release()
