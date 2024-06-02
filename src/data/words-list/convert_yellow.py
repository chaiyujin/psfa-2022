import math
import os

_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_DIR, "raw.txt")) as fp:
    words0 = {}
    words1 = {}

    def _insert(words, _id, txt):
        _id = _id.replace(".", "")
        words[int(_id)] = txt

    for line in fp:
        line = line.strip()
        if len(line) == 0:
            continue

        ss = line.split()
        assert len(ss) == 8
        _insert(words0, ss[0], ss[1])
        _insert(words0, ss[2], ss[3])

        _insert(words1, ss[4], ss[5])
        _insert(words1, ss[6], ss[7])


def _dict_to_list(words):
    return [words[i + 1] for i in range(len(words))]


words_list0 = _dict_to_list(words0)
words_list1 = _dict_to_list(words1)


def _print_table(list_id, words_list, n_cols, file=None):
    width = max(max(len(x) for x in words_list), 3)
    n_rows = int(math.ceil(len(words_list) / n_cols))

    def _print_line(txts):
        print("| " + " | ".join(str(x).center(width) for x in txts) + " |", file=file)

    print(f"## List {list_id} \n", file=file)
    _print_line(list(range(1, n_cols + 1)))
    _print_line([":" + "-" * (width - 2) + ":"] * n_cols)

    for r in range(n_rows):
        txts = []
        for c in range(n_cols):
            idx = c * n_rows + r
            if idx >= len(words_list):
                txt = ""
            else:
                txt = words_list[idx]
            txts.append(txt)
        _print_line(txts)
    print(file=file)


# _print_table(words_list0, 5)
# _print_table(words_list1, 5)

with open("words-PAL-PB50.md", "w") as fp:
    for i in range(1, 21):
        _print_table(i, ["     "] * 50, 5, file=fp)
