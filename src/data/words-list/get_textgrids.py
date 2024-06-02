import os
import warnings

import cv2
import librosa
import matplotlib

matplotlib.use("agg")
from shutil import copyfile, rmtree

import matplotlib.pyplot as plt
import numpy as np
from textgrids import TextGrid
from tqdm import tqdm

from src.engine.misc import filesys

warnings.filterwarnings("ignore")
_DICTIONARY = "./data/cmudict.txt"

# read dictionary
dictionary = dict()
with open(_DICTIONARY) as fp:
    for line in fp:
        line = line.strip()
        word = line.split()[0]
        if word not in dictionary:
            dictionary[word] = []
        dictionary[word].append(line)


def figure_to_numpy(fig) -> np.ndarray:
    # save it to a numpy array.
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_wav_phones(y, sr, text_grid):

    wd_list = []
    ph_list = []
    for syll in text_grid["words"]:
        label = syll.text.transcode()
        if len(label) == 0:
            label = "#"
        wd_list.append((label, syll.xmin, syll.xmax))
    for syll in text_grid["phones"]:
        label = syll.text.transcode()
        if len(label) == 0:
            label = "#"
        ph_list.append((label, syll.xmin, syll.xmax))

    # * plot
    vmin, vmax = min(y), max(y)
    vmin, vmax = -1, 1
    fig = plt.figure(figsize=(15, 5))
    plt.plot(y)
    plt.ylim(vmin, vmax)
    plt.xlim(0, len(y) - 1)
    plt.xticks([])
    for i, (label, x, _) in enumerate(ph_list):
        x = int(x * sr)
        y = vmax - 0.05 - 0.05 * (i % 5)
        plt.vlines(x=x, ymin=vmin, ymax=vmax, color="r", linestyles="--")
        plt.text(x, y, label, c="g", fontsize="small")
    for i, (label, x, _) in enumerate(wd_list):
        x = int(x * sr)
        y = vmax
        plt.text(x, y, label, c="b", fontsize="small")
    plt.tight_layout()
    img = figure_to_numpy(fig)[..., [2, 1, 0]]
    plt.close(fig)

    # cv2.imshow("img", img)
    # cv2.waitKey(1)
    return img


def align_each(root_dir, tmp_dir=".snaps/mfa/tmp"):
    dat_dir = os.path.join(tmp_dir, "dat")
    res_dir = os.path.join(tmp_dir, "out")
    if os.path.exists(dat_dir):
        rmtree(dat_dir)
    if os.path.exists(res_dir):
        rmtree(res_dir)
    os.makedirs(dat_dir, exist_ok=True)

    jobs = []
    filepaths = filesys.find_files(root_dir, r".*\.(mp3|wav|ogg)", recursive=True)
    for apath in filepaths:
        word = os.path.splitext(os.path.basename(apath))[0]
        if word[-3:] in ["-en", "-uk"]:
            word = word[:-3]

        tpath = os.path.join(os.path.dirname(apath), "textgrids", f"{word}.TextGrid")
        ipath = os.path.join(os.path.dirname(apath), "debug_align", f"{word}.png")
        os.makedirs(os.path.dirname(tpath), exist_ok=True)
        os.makedirs(os.path.dirname(ipath), exist_ok=True)

        # os.remove(os.path.splitext(apath)[0] + ".png")
        if os.path.exists(os.path.splitext(apath)[0] + ".TextGrid"):
            os.remove(os.path.splitext(apath)[0] + ".TextGrid")
        if os.path.exists(os.path.splitext(apath)[0] + "_align.png"):
            os.remove(os.path.splitext(apath)[0] + "_align.png")

        # * for mfa
        relpath = os.path.relpath(apath, root_dir).replace("/", "-")
        new_path = os.path.join(dat_dir, relpath)
        txt_path = os.path.splitext(new_path)[0] + ".txt"
        copyfile(apath, new_path)
        with open(txt_path, "w") as fp:
            fp.write(word)

        res_path = os.path.join(res_dir, os.path.splitext(relpath)[0] + ".TextGrid")
        jobs.append((word, apath, tpath, ipath, res_path))

    # * run
    cmd = (
        "mfa align"
        f" {dat_dir}"
        f" {_DICTIONARY} english"
        f" {res_dir}"
        f" -t {os.path.join(tmp_dir, 'tmp')} --clean --disable_mp"
    )
    os.system(cmd)

    for word, apath, tpath, ipath, res_path in tqdm(jobs):
        if os.path.exists(res_path):
            copyfile(res_path, tpath)
        else:
            print("Failed to find aligned .TextGrid: {}".format(word))
            continue

        # vis
        textgrid = TextGrid(tpath)
        y, sr = librosa.core.load(apath, 16000)
        im = plot_wav_phones(y, sr, textgrid)

        cv2.imwrite(ipath, im)
        cv2.imshow("img", im)
        cv2.waitKey(1)


if __name__ == "__main__":
    # align_each("./data/words-list/NU-6")
    # align_each("./data/words-list/PAL-PB50")
    # align_each("./data/words-list/MY")
    align_each("./data/words-list/WORDS_RAW")
