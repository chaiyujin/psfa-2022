import json
import os
import re

from tqdm import tqdm

from .download import CambridgeDictionary

_DIR = os.path.dirname(os.path.abspath(__file__))
REGEX_DIGIT = re.compile(r"\d")

DICTIONARY_PATH = "./data/cmudict.txt"
FREQUENT_WORDS_PATH = "~/Documents/Project2021/RelatedWorks/google-10000-english/google-10000-english-usa.txt"

BLACKLIST = ["BLOWJOB", "PUSSY", "AAA", "BRAD", "PTY", "VER", "NAV", "ADA", "DEC", "NOV", "TEL", "DEL"]

downloader = CambridgeDictionary(os.path.join(_DIR, "cambridgedictionary"))


PHS = []
with open(os.path.join(_DIR, "phones.txt")) as fp:
    for line in fp:
        line = line.strip()
        if len(line) == 0:
            continue
        PHS.append(line)
assert len(PHS) == 40


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

    word_lists = sum(word_lists, [])
    word_lists = [x.upper() for x in word_lists]
    return word_lists


def read_word_lists_from_txt(txt_path):
    words = []
    with open(txt_path) as fp:
        for line in fp:
            line = line.strip()
            if len(line) == 0 or line[0] == "'" or line[0] == "#":
                continue
            words.extend([x.lower() for x in line.split()])
    words = [x.upper() for x in words]
    return words


def read_dictionary():
    dictionary = dict()
    regex = re.compile(r"\(\d+\)")
    reword = re.compile(r"^[A-Z].*")
    with open(DICTIONARY_PATH) as fp:
        for line in fp:
            line = line.strip()
            word = line.split()[0]
            word = regex.sub("", word)
            if not reword.match(word):
                continue
            phs = line.split()[1:]
            if word not in dictionary:
                dictionary[word] = []
            dictionary[word].append(["#"] + phs + ["#"])
    return dictionary


def read_frequent_words():
    with open(os.path.expanduser(FREQUENT_WORDS_PATH)) as fp:
        frequent_words = set()
        for word in fp:
            word = word.strip().upper()
            if len(word) == 0:
                continue
            if word in BLACKLIST:
                print("Ignore black list word: {}".format(word))
                continue
            if len(word) <= 2:
                continue
            frequent_words.add(word)
    return frequent_words


def _key(phs):
    return " ".join(f"{REGEX_DIGIT.sub('', x):^3}" for x in phs)


def _get_biphones(pron):
    ret = []
    for i in range(len(pron) - 1):
        ret.append(_key([pron[i], pron[i + 1]]))
    return ret


# * Read Dictionary
dictionary = read_dictionary()

# * Read words
# word_candidates = read_frequent_words()

word_candidates = read_word_lists_from_md(os.path.join(_DIR, "words-NU-6.md"))
word_candidates += read_word_lists_from_md(os.path.join(_DIR, "words-PAL-PB50.md"))


# * Filter words
def _valid(word):
    if word not in dictionary:
        return False
    if word.find("'") >= 0 or word.find(".") >= 0 or word.find("-") >= 0:  # combination
        return False
    if word[-1] == "S" and word[:-1] in word_candidates:  # plural form
        return False
    return True


word_candidates = [x for x in word_candidates if _valid(x)]
# print(word_candidates)
# print(len(word_candidates))
# quit(1)

# * ----------------------------------------------- find all di-grams ---------------------------------------------- * #

BIPH_DICT = dict()
for x in PHS:
    for y in PHS:
        if x != y:
            key = _key([x, y])
            BIPH_DICT[key] = []

WORD_BI = dict()
for word in word_candidates:
    pron_list = dictionary[word]

    # check the pronunciations
    pron = pron_list[0]
    bi_list = _get_biphones(pron)
    for bi in bi_list:
        if bi not in BIPH_DICT:
            BIPH_DICT[bi] = []
        BIPH_DICT[bi].append(word)
    WORD_BI[word] = bi_list


# sort words
def sort_words():
    for key in BIPH_DICT:
        BIPH_DICT[key] = sorted(list(set(BIPH_DICT[key])), key=lambda x: f"{999-len(WORD_BI[x]):03d}{len(x):03d}-{x}")


sort_words()

# * ----------------------------------------------- prepare to select ---------------------------------------------- * #

selected = set()
selected_biphones = dict()
missing_biphones = set()


def _select(word, pron):
    if word in selected:
        return

    print("select", word)
    selected.add(word)
    bis = _get_biphones(pron)
    for biphone in bis:
        if biphone not in selected_biphones:
            selected_biphones[biphone] = list()
        selected_biphones[biphone].append(word)


failure = set()
all_bigrams = sorted(list(BIPH_DICT.keys()))

n_per = 1
progress = tqdm(range(len(all_bigrams) * n_per))
while True:
    missing = [
        key
        for key in all_bigrams
        if ((key not in selected_biphones or len(selected_biphones[key]) < n_per) and not key in missing_biphones)
    ]
    if len(missing) == 0:
        break

    done = False
    for key in missing:
        words = [x for x in BIPH_DICT[key] if (x not in selected and x not in failure)]
        for word in words:  # already sorted
            if not downloader.download(word.lower()):
                failure.add(word)
                BIPH_DICT[key].remove(word)
                print("Failed to download '{}'".format(word))
                continue
            _select(word, dictionary[word][0])
            BIPH_DICT[key].remove(word)
            done = True
            break
        if done:
            break

    if not done:
        for k in missing:
            for x in BIPH_DICT[k]:
                assert x in selected or x in failure
            missing_biphones.add(k)
    progress.update()
progress.close()

# * Print selected results
biphones = sorted(list(selected_biphones.keys()))
count_selected = 0
for biph in biphones:
    words = selected_biphones[biph]
    print(biph, len(words), "|", " ".join(x.lower() for x in words))
    count_selected += 1
count_missing = 0
for bi in BIPH_DICT:
    if bi not in selected_biphones:
        # print(bi, "is not selected")
        count_missing += 1
print(count_selected, count_missing, len(all_bigrams))
print(len(selected))

# * For each di-gram, select
with open(os.path.join(_DIR, "words_raw.txt"), "w") as fp:
    for word in sorted(list(selected)):
        print(word, file=fp)

# groups = dict()
# for k, v in selected_biphones.items():
#     ss = k.split()
#     key = "-".join(ss)
#     groups[key] = [v[0].lower()]

# with open(os.path.join(_DIR, "words.json"), "w") as fp:
#     fp.write("{\n")
#     fp.write("  \"group\": {\n")
#     all_keys = sorted(list(groups.keys()))
#     for i, key in enumerate(all_keys):
#         words_txt = ",".join(f"\"{x}\"" for x in groups[key])
#         fp.write(f"    \"{key}\"{' '*(5-len(key))}: [{words_txt}]")
#         if i != len(all_keys) - 1:
#             fp.write(",\n")
#         else:
#             fp.write("\n")
#     fp.write("  }\n")
#     fp.write("}\n")


DOWN_DIR = os.path.join(downloader.save_dir, "success")
SAVE_DIR = os.path.join(_DIR, "WORDS_RAW")
os.makedirs(os.path.join(SAVE_DIR), exist_ok=True)
for word in tqdm(sorted(selected)):
    fname = word.lower() + ".mp3"
    new_fname = word.lower() + ".mp3"
    in_file = os.path.join(DOWN_DIR, fname)
    out_file = os.path.join(SAVE_DIR, new_fname)
    if os.path.exists(out_file):
        continue

    cmd = f"ffmpeg -loglevel error -i '{in_file}' -ac 1 -af 'adelay=8000S,apad=pad_len=4000' '{out_file}' -y"
    # cmd = f"ffmpeg -loglevel error -i '{in_file}' -ac 1  '{out_file}' -y"
    os.system(cmd)

    import cv2
    import librosa
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    from src.engine.painter import figure_to_numpy

    y, sr = librosa.core.load(out_file)
    fig = plt.figure()
    plt.plot(y)
    im = figure_to_numpy(fig)
    plt.close(fig)
    cv2.imshow("img", im)
    cv2.waitKey(1)
