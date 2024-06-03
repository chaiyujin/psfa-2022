import os
from functools import lru_cache

import numpy as np

from assets import DATASET_ROOT, FACE_LOWER_VIDX, FACE_NEYE_VIDX, LIPS_VIDX
from src.data.mesh import load_mesh
from src.engine.misc import filesys
from src.engine.painter import figure_to_numpy


def is_special(y):
    return False


def tsne_visualization(z_list, y_list, n_ids=None, file_path=None):
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt
    from MulticoreTSNE import MulticoreTSNE as TSNE

    z_list = np.asarray(z_list)
    y_list = np.asarray(y_list)

    # find for same y
    y_dict = dict()
    set_y = set()
    all_y = []
    for i, y in enumerate(y_list):
        if y not in y_dict:
            y_dict[y] = []
        y_dict[y].append(i)
        if y not in set_y:
            set_y.add(y)
            all_y.append(y)

    # print("PCA: do pca of {} data...".format(len(y_list)), end="\n")
    # pca = PCA(n_components=10)
    # z_list = pca.fit_transform(z_list)

    print("TSNE: visualizing {} data...".format(len(y_list)), end="\n")
    tsne = TSNE(n_jobs=4)
    data = tsne.fit_transform(z_list)
    fig = plt.figure(figsize=(5, 5))
    for y in reversed(all_y):
        indices = y_dict[y]
        plt.scatter(data[indices, 0], data[indices, 1], label=y, marker="o" if is_special(y) else ".", alpha=0.5)
    plt.legend()
    plt.axis("off")
    plt.tight_layout()
    img = None
    if file_path is not None:
        if len(os.path.dirname(file_path)) > 0:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if os.path.splitext(file_path)[1] != ".png":
            file_path += ".png"
        plt.savefig("{}".format(file_path), format="png")
    else:
        img = figure_to_numpy(fig)
        plt.show()
    plt.close(fig)
    print("Done!")
    return img


@lru_cache(maxsize=1)
def _load_training_offsets(speaker):
    spk_dir = f"{DATASET_ROOT}/talk_video/celebtalk/data/{speaker}/fitted"
    idle, tris, _ = load_mesh(spk_dir + "/identity/identity.obj")

    offsets = []
    seq_dirs = filesys.find_dirs(spk_dir, r"(trn|vld)-\d\d\d", recursive=False)
    for seq_dir in seq_dirs:
        meshes_dir = seq_dir + "/meshes"
        meshes_npy = filesys.find_files(meshes_dir, r"\d{6}\.npy", recursive=False)
        for mesh_npy in meshes_npy:
            vert = np.load(mesh_npy)
            offsets.append(vert - idle)
    offsets = np.asarray(offsets, dtype=np.float32)
    return offsets


def _load_results_offsets(epoch):
    prefix = f"media/[{epoch}][test]"

    offsets = []
    npy_path = prefix + "vocaset/FaceTalk_170811_03274_TA/sentence03/dump-offsets-final.npy"
    offsets.append(np.load(npy_path))

    # npy_path0 = prefix + f"{speaker}/vld-000@style_trn-000/dump-offsets-final.npy"
    # npy_path1 = prefix + f"{speaker}/vld-001@style_trn-001/dump-offsets-final.npy"
    # offsets = np.concatenate((np.load(npy_path0), np.load(npy_path1)))

    # for npy_path in filesys.find_files(prefix + "test-clips", r"dump-offsets-final\.npy"):
    #     offsets.append(np.load(npy_path)[::3])

    offsets = np.concatenate(offsets)
    return offsets


def do_tsne(epoch, speaker):
    decmp_data = _load_results_offsets(epoch)
    decmp_lbls = ["decmp"] * len(decmp_data)

    train_data = _load_training_offsets(speaker)
    train_lbls = ["train"] * len(train_data)

    data = np.concatenate((decmp_data, train_data))
    lbls = decmp_lbls + train_lbls

    data = data[:, FACE_LOWER_VIDX, :]
    data = np.reshape(data, (data.shape[0], -1))
    tsne_visualization(data, lbls, file_path=f"media/tsne-epoch{epoch}.png")


# RUNS_ROOT = "runs/face_noeyeballs"
# best_epochs = dict(
#     m001_trump=dict(
#         track=158,
#         cmb3d=96,
#         decmp=100,
#         vocaft=173,
#     ),
#     f000_watson=dict(
#         track=60,
#         cmb3d=60,
#         decmp=60,
#         vocaft=40,
#     ),
# )


# for speaker in ["m001_trump", "f000_watson"]:

#     track_data = _load_results_offsets(speaker, "track")
#     track_lbls = ["gen_track"] * len(track_data)

#     cmb3d_data = _load_results_offsets(speaker, "cmb3d")
#     cmb3d_lbls = ["gen_cmb3d"] * len(cmb3d_data)

#     decmp_data = _load_results_offsets(speaker, "decmp")
#     decmp_lbls = ["gen_decmp"] * len(decmp_data)

#     data = np.concatenate((train_data, track_data, cmb3d_data, decmp_data))
#     lbls = train_lbls + track_lbls + cmb3d_lbls + decmp_lbls

#     data = data[:, FACE_LOWER_VIDX, :]
#     print(data.shape)
#     data = np.reshape(data, (data.shape[0], -1))
#     tsne_visualization(data, lbls, file_path=f"{RUNS_ROOT}/{speaker}_tsne.png")
