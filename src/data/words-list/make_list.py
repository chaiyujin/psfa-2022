import os
import warnings

import librosa
import matplotlib
import numpy as np
import torch
import torchaudio as ta

matplotlib.use("agg")
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.audio.vad import detect_speech, to_pairs
from src.engine.painter import figure_to_numpy

warnings.filterwarnings("ignore")
_DIR = os.path.dirname(os.path.abspath(__file__))


def read_word_lists_from_md(md_fpath):
    word_lists = []
    words = []
    with open(md_fpath) as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0 or line[0] != "|":
                continue
            line = [x.strip() for x in line.split("|")[1:-1]]
            if line[0].startswith("0") or line[0].startswith("1"):
                continue
            if line[0].startswith(":") or line[0].startswith("-"):
                if len(words) > 0:
                    word_lists.append(words)
                words = []
                continue
            words.extend([x.lower() for x in line])
    if len(words) > 0:
        word_lists.append(words)
    return word_lists


def read_word_lists_from_txt(txt_path):
    words = []
    with open(txt_path) as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0 or line[0] == "'" or line[0] == "#":
                continue
            words.extend([x.lower() for x in line.split()])
    return [words]


def export_audio_files(src_dir, out_dir, word_lists):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "debug"), exist_ok=True)

    for i_list, word_list in enumerate(word_lists):
        # out_dir_list = os.path.join(out_dir, f"list{i_list}")

        for word in tqdm(word_list):
            # if word not in ["aurora", "beijing"]:
            #     continue

            src_apath = os.path.join(src_dir, f"{word}.mp3")
            dst_apath = os.path.join(out_dir, f"{word}.mp3")
            dbg_ipath = os.path.join(out_dir, "debug", f"{word}.png")
            if not os.path.exists(src_apath):
                continue
            if os.path.exists(dst_apath):
                continue

            y, sr = librosa.core.load(src_apath, sr=48000)
            vad = detect_speech(y, sr, smooth_ms=10, vad_mode=3)
            pairs = to_pairs(vad)
            assert len(pairs) >= 1
            stt = pairs[0][0]
            end = pairs[-1][-1]
            assert end > stt

            prev = sr // 5 - stt
            post = sr // 10 - (len(y) - end)
            if prev < 0:
                y = y[-prev:]
                vad = vad[-prev:]
            else:
                y = np.pad(y, [prev, 0], "constant")
                vad = np.pad(vad, [prev, 0], "constant")
            if post < 0:
                y = y[:post]
                vad = vad[:post]
            else:
                y = np.pad(y, [0, post], "constant")
                vad = np.pad(vad, [0, post], "constant")

            fig, axes = plt.subplots(2, sharex=True)
            plt.title(word)
            axes[0].plot(y)
            axes[1].plot(vad)
            img = figure_to_numpy(fig)
            plt.close(fig)
            cv2.imwrite(dbg_ipath, img)
            cv2.imshow("img", img)
            cv2.waitKey(1)

            ta.save(dst_apath, torch.tensor(y[None, ...]), sr)


if __name__ == "__main__":
    # src_dir = os.path.join(_DIR, "macmillandictionary", "success")
    src_dir = os.path.join(_DIR, "cambridgedictionary", "success")

    # out_dir = os.path.join(_DIR, "PAL-PB50")
    # word_lists = read_word_lists_from_md(os.path.join(_DIR, "words-PAL-PB50.md"))
    # export_audio_files(src_dir, out_dir, word_lists)

    # out_dir = os.path.join(_DIR, "MY")
    # word_lists = read_word_lists_from_txt(os.path.join(_DIR, "words-MY.txt"))
    # export_audio_files(src_dir, out_dir, word_lists)

    out_dir = os.path.join(_DIR, "WORDS_RAW")
    word_lists = read_word_lists_from_txt(os.path.join(_DIR, "words_raw.txt"))
    export_audio_files(src_dir, out_dir, word_lists)
