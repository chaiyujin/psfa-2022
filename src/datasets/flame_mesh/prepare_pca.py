import logging
import os

import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.datasets.vocaset import string_of_ignored_speakers
from src.modules.flame_utils import get_flame_utils
from src.utils.data import image as imutils
from src.utils.data.mesh import load_mesh
from src.utils.data.mesh import viewer as mesh_viewer
from src.utils.data.video import VideoWriter
from src.utils.misc import filesys
from src.utils.painter import Text, put_texts

logger = logging.getLogger(__name__)


def _get_speaker(x):
    spk = os.path.basename(os.path.dirname(os.path.dirname(x)))
    assert spk.startswith("FaceTalk_")
    return spk


def _debug(title, offsets):
    tmpl_verts = get_flame_utils().template_vertices("FULL", "", True)
    verts = tmpl_verts + np.reshape(offsets, (-1, 3))
    im = mesh_viewer.render(verts)
    imutils.imshow(title, im)
    imutils.waitKey(1)


def do_pca(config, output_subdir, subdirs, scaled=False, eyes_split=False, ignore_speakers=None):
    # * Log information
    logger.info("Do PCA in root: '{}'".format(config.root))
    logger.info("  For subdirs: {}".format(subdirs))
    if ignore_speakers is not None and len(ignore_speakers) > 0:
        logger.info("  Ignore Speakers: {}".format(",".join(x for x in ignore_speakers)))
        # to set for better speed
        ignore_speakers = set(ignore_speakers)

    # * Save paths for PCA
    pca_dir = os.path.join(output_subdir, string_of_ignored_speakers(ignore_speakers))
    if scaled:
        pca_dir += "-scaled"
    if eyes_split:
        pca_dir += "-split"
    path_comp = os.path.join(config.root, pca_dir, "offsets_comp.npy")
    path_mean = os.path.join(config.root, pca_dir, "offsets_mean.npy")
    path_vars = os.path.join(config.root, pca_dir, "offsets_vars.npy")
    path_rate = os.path.join(config.root, pca_dir, "offsets_rate.npy")
    logger.info("  Into subdir: {}".format(pca_dir))

    # * Find input files
    file_list = []
    for subdir in subdirs:
        dirpath = os.path.join(config.root, subdir)
        files = filesys.find_files(dirpath, r"offsets\.npy", True, True)
        file_list.extend(files)
    # Filter speakers
    if ignore_speakers is not None and len(ignore_speakers) > 0:
        file_list = [x for x in file_list if _get_speaker(x) not in ignore_speakers]

    # for x in file_list:
    #     print(x)
    # print(len(file_list))
    # quit()

    # * Init debug viewer
    idle_path = os.path.join(os.path.dirname(file_list[0]), "idle.obj")
    mesh_viewer.set_template(filename=idle_path)

    # * Do PCA
    if not os.path.exists(path_comp):
        if not eyes_split:
            _normally_pca(config, file_list, scaled, path_comp, path_mean, path_vars, path_rate)
        else:
            _two_parts_pca(config, file_list, scaled, path_comp, path_mean, path_vars, path_rate)
        logger.info("PCA is done!")
    else:
        logger.info("PCA is cached!")

    # * Load the PCA
    comp = np.load(path_comp)
    mean = np.load(path_mean)
    stdv = np.sqrt(np.load(path_vars))
    # rate = np.load(path_rate)
    # logger.info("PCA first 50 compoenents got {:.1f}% energy".format(sum(rate[:50]) * 100))
    # print(comp.shape, stdv.shape, stdv)

    # Check orth
    C = np.matmul(comp.T, comp)
    assert np.isclose(C, np.eye(C.shape[0]), atol=1e-4).all()

    # * Visualize components
    idle, _, _ = load_mesh(idle_path)
    dir_video = os.path.join(config.root, pca_dir, "offsets_videos")
    for i in tqdm(range(min(comp.shape[1], 50))):
        vpath = os.path.join(dir_video, f"comp{i:02d}.mp4")
        if os.path.exists(vpath):
            continue
        writer = VideoWriter(output_path=vpath, fps=30, high_quality=True, makedirs=True)
        coeffs = np.zeros((comp.shape[1]), dtype=comp.dtype)
        for factor in (
            list(np.linspace(0, 3, num=30))
            + list(np.linspace(3, 0, num=30))
            + list(np.linspace(0, -3, num=30))
            + list(np.linspace(-3, 0, num=30))
        ):
            coeffs[i] = factor * stdv[i]
            d = np.dot(comp, coeffs[:, None])[:, 0] + mean
            d = np.reshape(d, (-1, 3))
            verts = idle + d
            img = mesh_viewer.render(verts, dtype=np.uint8)
            txt = Text(
                f"Comp {i:02d}: {coeffs[i]:9.6f} (x {factor:6.3f} std)", (img.shape[1] // 2, 0), ("center", "top")
            )
            img = put_texts(img, [txt], font_size=18)
            writer.write_image(img)
            # imutils.imshow("img", img)
            # imutils.waitKey(1)
        writer.close()


def _normally_pca(config, file_list, scaled, path_comp, path_mean, path_vars, path_rate):
    vidx = get_flame_utils().vidx("FULL", "", True)
    vidx_sym = [get_flame_utils().symmetric_index(x) for x in vidx]

    os.makedirs(os.path.dirname(path_comp), exist_ok=True)

    if config.get("debug"):
        file_list = file_list[:10]

    data_list = []
    for i, fpath in enumerate(tqdm(file_list, desc="Loading data")):
        data = np.load(fpath)
        # for vocaset, we downsample fps
        if fpath.find("data_vocaset") >= 0:
            data = data[::2, ...]
        data_list.append(data)
        # symmetry
        data_sym = data.copy()
        data_sym[:, vidx, :] = data[:, vidx_sym, :] * np.asarray([[[-1, 1, 1]]])
        data_list.append(data_sym)

        # for off, off_sym in zip(data, data_sym):
        #     _debug('img', off)
        #     _debug('img', off_sym)
    X = np.concatenate(data_list, axis=0)
    X = np.reshape(X, (X.shape[0], -1))

    logger.info("PCA {} data...".format(len(X)))
    pca = PCA(n_components=300, copy=False, whiten=False)
    pca.fit(X)

    if not scaled:
        np.save(path_vars, pca.explained_variance_)
        np.save(path_rate, pca.explained_variance_ratio_)
        np.save(path_mean, pca.mean_)
        np.save(path_comp, pca.components_.T)
    else:
        np.save(path_vars, np.ones_like(pca.explained_variance_))
        np.save(path_rate, pca.explained_variance_ratio_)
        np.save(path_mean, pca.mean_)
        np.save(path_comp, pca.components_.T * np.sqrt(pca.explained_variance_)[None, :])


def _two_parts_pca(config, file_list, scaled, path_comp, path_mean, path_vars, path_rate):
    assert not scaled
    vidx = get_flame_utils().vidx("FULL", "", True)
    vidx_sym = [get_flame_utils().symmetric_index(x) for x in vidx]
    vidx_above = get_flame_utils().vidx("FULL", "EYES_ABOVE", True)
    vidx_below = [i for i in range(5023) if i not in vidx_above]

    os.makedirs(os.path.dirname(path_comp), exist_ok=True)

    if config.get("debug"):
        file_list = file_list[:10]

    part0_data_list = []
    part1_data_list = []

    def _append_data(data):
        part0 = data.copy()
        part1 = data.copy()
        part0[:, vidx_below, :] = 0  # eyes_above part, set below as 0
        part1[:, vidx_above, :] = 0
        part0_data_list.append(part0)
        part1_data_list.append(part1)

        # for off0, off1 in zip(part0, part1):
        #     _debug('part0', off0)
        #     _debug('part1', off1)

    for i, fpath in enumerate(tqdm(file_list, desc="Loading data")):
        data = np.load(fpath)
        # for vocaset, we downsample fps
        if fpath.find("data_vocaset") >= 0:
            data = data[::2, ...]
        _append_data(data)

        # symmetry
        data_sym = data.copy()
        data_sym[:, vidx, :] = data[:, vidx_sym, :] * np.asarray([[[-1, 1, 1]]])
        _append_data(data_sym)

    def _save_pca_for_part(part, pca):
        np.save(os.path.splitext(path_comp)[0] + f"-{part}.npy", pca.components_.T)
        np.save(os.path.splitext(path_mean)[0] + f"-{part}.npy", pca.mean_)
        np.save(os.path.splitext(path_vars)[0] + f"-{part}.npy", pca.explained_variance_)
        np.save(os.path.splitext(path_rate)[0] + f"-{part}.npy", pca.explained_variance_ratio_)

    # * Above eyes
    if not os.path.exists(os.path.splitext(path_rate)[0] + "-above_eyes.npy"):
        X0 = np.concatenate(part0_data_list, axis=0)
        X0 = np.reshape(X0, (X0.shape[0], -1))
        logger.info("PCA {} data for part above eyes...".format(len(X0)))
        pca0 = PCA(n_components=300, copy=False, whiten=False)
        pca0.fit(X0)
        _save_pca_for_part("above_eyes", pca0)

    # * Below eyes
    if not os.path.exists(os.path.splitext(path_rate)[0] + "-below_eyes.npy"):
        X1 = np.concatenate(part1_data_list, axis=0)
        X1 = np.reshape(X1, (X1.shape[0], -1))
        logger.info("PCA {} data for part below eyes...".format(len(X1)))
        pca1 = PCA(n_components=300, copy=False, whiten=False)
        pca1.fit(X1)
        _save_pca_for_part("below_eyes", pca1)

    # * Merge two parts
    part0_comp = np.load(os.path.splitext(path_comp)[0] + "-above_eyes.npy")
    part0_mean = np.load(os.path.splitext(path_mean)[0] + "-above_eyes.npy")
    part0_vars = np.load(os.path.splitext(path_vars)[0] + "-above_eyes.npy")
    part0_rate = np.load(os.path.splitext(path_rate)[0] + "-above_eyes.npy")
    part1_comp = np.load(os.path.splitext(path_comp)[0] + "-below_eyes.npy")
    part1_mean = np.load(os.path.splitext(path_mean)[0] + "-below_eyes.npy")
    part1_vars = np.load(os.path.splitext(path_vars)[0] + "-below_eyes.npy")
    part1_rate = np.load(os.path.splitext(path_rate)[0] + "-below_eyes.npy")

    # !HACK: hard-coded number of above eyes components
    # part0_rate_acc = []
    # for i, x in enumerate(part0_rate):
    #     if i == 0:
    #         part0_rate_acc.append(x)
    #     else:
    #         part0_rate_acc.append(x + part0_rate_acc[i-1])

    n_above = 6
    m_comp = np.concatenate((part0_comp[:, :n_above], part1_comp[:, :-n_above]), axis=-1)
    m_vars = np.concatenate((part0_vars[:n_above], part1_vars[:-n_above]), axis=-1)
    m_mean = part0_mean + part1_mean

    np.save(path_vars, m_vars)
    np.save(path_mean, m_mean)
    np.save(path_comp, m_comp)
