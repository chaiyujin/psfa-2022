import json
import os
import re
from shutil import copyfile, copytree

import textgrids

from .make_list import read_word_lists_from_md, read_word_lists_from_txt

_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_all_words(dat_dir, word_lists):
    regex_digit = re.compile(r"\d+")
    all_words = dict()
    for i_list, word_list in enumerate(word_lists):
        for word in word_list:
            apath = os.path.join(dat_dir, f"list{i_list}", f"{word}.mp3")
            if not os.path.exists(apath):
                apath = os.path.join(dat_dir, f"{word}.mp3")
            tpath = os.path.splitext(apath)[0] + ".TextGrid"
            if not os.path.exists(tpath):
                tpath = os.path.join(os.path.dirname(apath), "textgrids", f"{word}.TextGrid")
            if not os.path.exists(apath) or not os.path.exists(tpath):
                continue

            textgrid = textgrids.TextGrid(tpath)
            ph_list = []
            for syll in textgrid["phones"]:
                label = syll.text.transcode()
                if len(label) == 0:
                    label = "#"
                label = regex_digit.sub("", label)
                ph_list.append((label, syll.xmin, syll.xmax))
            all_words[word] = ph_list
    return all_words


def _dump_dict(data, fp, indent, is_pron=False):
    all_keys = sorted(list(data.keys()))
    max_key_len = max(len(x) for x in all_keys)
    for i, key in enumerate(all_keys):
        if i > 0:
            fp.write(",\n")
        fp.write(" " * indent + f'"{key}"{" " * (max_key_len-len(key))}: ')
        if not is_pron:
            fp.write(json.dumps(data[key]) + "")
        else:
            txt = "["
            for i, tup in enumerate(data[key]):
                if i > 0:
                    txt += ", "
                txt += "["
                txt += f'"{tup[0]}",{" " * (2-len(tup[0]))}'
                txt += f" {tup[1]:5.3f},"
                txt += f" {tup[2]:5.3f}"
                txt += "]"
            txt += "]"
            fp.write(txt)
            pass
    fp.write("\n")


def _analyze_groups(all_words):
    def _key(*phs):
        return "-".join(x for x in phs)

    def _get_biphones(org_pron):
        # ! for diphthongs, we split it into two vowels
        pron = []
        for ph in org_pron:
            ph = ph[0]
            pron.append(ph)
            # if ph == "AW":
            #     pron.extend(["AA", "UH"])
            # elif ph == "AY":
            #     pron.extend(["AA", "IH"])
            # elif ph == "EY":
            #     pron.extend(["EH", "IH"])
            # elif ph == "OW":
            #     pron.extend(["AO", "UH"])
            # elif ph == "OY":
            #     pron.extend(["AO", "IH"])
            # else:
            #     pron.append(ph)

        ret = []
        for i in range(len(pron) - 1):
            ret.append(_key(pron[i], pron[i + 1]))
        return ret

    bi_dict = dict()
    for word, pron in all_words.items():
        biphs = _get_biphones(pron)
        for key in biphs:
            if key not in bi_dict:
                bi_dict[key] = []
            bi_dict[key].append(word)

    # * select the best words
    for key in bi_dict:
        bi_dict[key] = sorted(bi_dict[key])
        bi_dict[key] = bi_dict[key][:1]  # ! HACK: only 1 word
    return bi_dict


if __name__ == "__main__":
    # dat_dir = os.path.join(_DIR, "NU-6")
    # word_lists = read_word_lists_from_md(os.path.join(_DIR, "words-NU-6.md"))
    # all_words = _load_all_words(dat_dir, word_lists)

    # dat_dir = os.path.join(_DIR, "PAL-PB50")
    # word_lists = read_word_lists_from_md(os.path.join(_DIR, "words-PAL-PB50.md"))
    # all_words_pb50 = _load_all_words(dat_dir, word_lists)

    # dat_dir = os.path.join(_DIR, "MY")
    # word_lists = read_word_lists_from_txt(os.path.join(_DIR, "words-MY.txt"))
    # all_words = _load_all_words(dat_dir, word_lists)

    dat_dir = os.path.join(_DIR, "WORDS_RAW")
    word_lists = read_word_lists_from_txt(os.path.join(_DIR, "words_raw.txt"))
    all_words = _load_all_words(dat_dir, word_lists)

    if not os.path.exists(os.path.join(_DIR, "words.json")):
        groups = _analyze_groups(all_words)
        words_sel = []
        for key, words in groups.items():
            words_sel.extend(words)
        words_sel = set(words_sel)
        words_dict = {k: v for k, v in all_words.items() if k in words_sel}
        print(len(words_sel), len(groups))

        out_dir = os.path.join(_DIR, "WORDS")
        os.makedirs(out_dir, exist_ok=True)
        if not os.path.isdir(os.path.join(out_dir, "debug_align")):
            copytree(os.path.join(dat_dir, "debug_align"), os.path.join(out_dir, "debug_align"))
        if not os.path.isdir(os.path.join(out_dir, "textgrids")):
            copytree(os.path.join(dat_dir, "textgrids"), os.path.join(out_dir, "textgrids"))
        for word in words_sel:
            copyfile(os.path.join(dat_dir, f"{word}.mp3"), os.path.join(out_dir, f"{word}.mp3"))

        with open(os.path.join(_DIR, "words.txt"), "w") as fp:
            for word in sorted(list(words_sel)):
                print(word, file=fp)

        with open(os.path.join(_DIR, "words.json"), "w") as fp:
            fp.write("{\n")

            fp.write('  "group": {\n')
            _dump_dict(groups, fp, 4)
            fp.write("  },\n")

            fp.write('  "pron": {\n')
            _dump_dict(words_dict, fp, 4, is_pron=True)
            fp.write("  }\n")

            fp.write("}\n")

        copyfile(os.path.join(_DIR, "words.json"), "benchmark/json/words.json")
        copyfile(os.path.join(_DIR, "words.json"), "benchmark_cmp/json/words.json")

    else:
        print("Found", os.path.join(_DIR, "words.json"), "Skip!")
