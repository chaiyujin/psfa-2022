import os
import re

import matplotlib.pyplot as plt
import seaborn
from tqdm import tqdm

from .download import CambridgeDictionary

_DIR = os.path.dirname(os.path.abspath(__file__))
REGEX_DIGIT = re.compile(r"\d")
DICTIONARY_PATH = "./data/cmudict.txt"
FREQUENT_WORDS_PATH = "~/Documents/Project2021/RelatedWorks/google-10000-english/google-10000-english-usa.txt"


def _simplify(ph):
    return f"{REGEX_DIGIT.sub('', ph):^3}"


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


def read_harvard_sentences(md_path):
    sentences = []
    with open(md_path) as fp:
        for line in fp:
            line = line.strip()
            if re.match(r"\d+\. .*", line):
                line = re.sub(r"\d+\. ", "", line)
                line = re.sub(r"[\.,]", "", line)
                line = line.upper()
                sentences.append(line)
    words = []
    for sent in sentences:
        words.extend(sent.split())
    return words


# * ------------------------------------------------ read dictionary ----------------------------------------------- * #

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

# * ---------------------------------------------- read frequent words --------------------------------------------- * #

with open(os.path.expanduser(FREQUENT_WORDS_PATH)) as fp:
    frequent_words = set()
    for word in fp:
        word = word.strip().upper()
        if len(word) == 0:
            continue
        if word not in dictionary:
            print("Failed to find {}".format(word))
            continue
        frequent_words.add(word)


frequent_words = read_harvard_sentences(os.path.join(_DIR, "harvard_sentences.md"))

# words_pb50 = sum(read_word_lists_from_md(os.path.join(_DIR, "words-PAL-PB50.md")), [])
# words_pb50 = [x.upper() for x in words_pb50]

# words_nu6 = sum(read_word_lists_from_md(os.path.join(_DIR, "words-NU-6.md")), [])
# words_nu6 = [x.upper() for x in words_nu6]

words = sum(read_word_lists_from_txt(os.path.join(_DIR, "words.txt")), [])
words = [x.upper() for x in words]
# print(words_test)

# * ----------------------------------------------- find all di-grams ---------------------------------------------- * #


def _get_frequency(words):

    ignore = 0
    PH_DICT = dict()
    for word in words:
        if word not in dictionary:
            ignore += 1
            continue

        pron_list = dictionary[word]

        # check the pronunciations
        pron = pron_list[0]
        for ph in pron:
            key = _simplify(ph)
            if key not in PH_DICT:
                PH_DICT[key] = set()
            PH_DICT[key].add(word)

    # with open("phones.txt", "w") as fp:
    #     for x in sorted(list(PH_DICT.keys())):
    #         fp.write(x + "\n")

    print("Total", len(words), "ignore", ignore)

    x = sorted([k for k in PH_DICT.keys()][1:])
    y = sum([[k] * len(PH_DICT[k]) for i, k in enumerate(x)], [])
    return y, x


plt.figure(figsize=(12, 5))
y, x = _get_frequency(frequent_words)
seaborn.histplot(data=y, stat="probability", bins=len(x), label="Harvard Sentences")
y, x = _get_frequency(words)
seaborn.histplot(data=y, stat="probability", bins=len(x), label="words", color="orange")
# y, x = _get_frequency(words_nu6)
# seaborn.histplot(data=y, stat='probability', bins=len(x), label='words nu6', color='red')
plt.legend()
plt.title("Frequency of Phonemes (ARPAbet)")
plt.tight_layout()
plt.savefig(os.path.join(_DIR, "words_freq.png"))
plt.show()
