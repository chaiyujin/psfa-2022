import math
import re

_phonemes = [
    "#",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]

_visemes = [
    "SIL",
    "AHH",
    "AEE",
    "IHH",
    "OHH",
    "UHH",
    "RRR",
    "WWW",
    "SHCH",
    "KKK",
    "THDH",
    "LNST",
    "FFF",
    "MBP",
]

_visemes_to_id = {v: i for i, v in enumerate(_visemes)}
_phonemes_to_id = {v: i for i, v in enumerate(_phonemes)}

_ph_to_vi = {
    # silence
    "#": "SIL",
    "SIL": "SIL",
    # AHH
    "AA": "AHH",
    "AH": "AHH",
    "ER": "AHH",
    # AEE
    "AE": "AEE",
    "EH": "AEE",
    # IHH
    "IH": "IHH",
    "IY": "IHH",
    "Y": "IHH",
    # OHH
    "AO": "OHH",
    # UHH
    "UH": "UHH",
    "UW": "UHH",
    # R, W
    "R": "RRR",
    "W": "WWW",
    # SHCH
    "SH": "SHCH",
    "ZH": "SHCH",
    "CH": "SHCH",
    "JH": "SHCH",
    # KKK
    "K": "KKK",
    "G": "KKK",
    "NG": "KKK",
    # THDH
    "TH": "THDH",
    "DH": "THDH",
    # LNST
    "L": "LNST",
    "N": "LNST",
    "T": "LNST",
    "D": "LNST",
    "S": "LNST",
    "Z": "LNST",
    # FFF
    "F": "FFF",
    "V": "FFF",
    # MBP
    "M": "MBP",
    "B": "MBP",
    "P": "MBP",
}
_di_to_vis = {
    "AW": ["AHH", "UHH"],
    "AY": ["AHH", "IHH"],
    "EY": ["AEE", "IHH"],
    "OW": ["OHH", "UHH"],
    "OY": ["OHH", "IHH"],
}

for ph in _phonemes:
    assert ph in _ph_to_vi or ph in _di_to_vis or ph in ["HH"]
for _, vi in _ph_to_vi.items():
    assert vi in _visemes

regex_digit = re.compile(r"\d+")


def viseme_to_id(vi):
    return _visemes_to_id[vi]


def id_to_viseme(_id):
    return _visemes[int(_id)]


def phoneme_to_id(ph):
    ph = regex_digit.sub("", ph)
    return _phonemes_to_id[ph]


def id_to_phoneme(_id):
    return _phonemes[int(_id)]


def is_silence(ph_or_vi):
    return ph_or_vi in ["#", "SIL", "v_SIL"]


def remove_stress(ph):
    return regex_digit.sub("", ph)


def is_vocalic(ph_or_vi):
    ph_or_vi = regex_digit.sub("", ph_or_vi)
    return ph_or_vi in [
        "AA",
        "AHH",
        "AH",
        "AHH",
        "ER",
        "AHH",
        "AE",
        "AEE",
        "EH",
        "AEE",
        "IH",
        "IHH",
        "IY",
        "IHH",
        "AO",
        "OHH",
        "UH",
        "UHH",
        "UW",
        "UHH",
        "AW",
        "AY",
        "EY",
        "OW",
        "OY",
    ]


def phonemes_to_visemes(ph_list):
    ret = []
    i = 0
    ph_list = [regex_digit.sub("", x) for x in ph_list]
    while i < len(ph_list):
        # * simple case
        if ph_list[i] in _ph_to_vi:
            ret.append(_ph_to_vi[ph_list[i]])
            i += 1
        # * HH
        elif ph_list[i] == "HH":
            j = i + 1
            while j < len(ph_list) and ph_list[j] == ph_list[i]:
                j += 1
            rep = None
            if j < len(ph_list):
                if ph_list[j] in _ph_to_vi:
                    rep = _ph_to_vi[ph_list[j]]
                else:
                    rep = _di_to_vis[ph_list[j]][0]
            else:
                rep = ret[-1] if len(ret) > 0 else "SIL"
            for _ in range(i, j):
                ret.append(rep)
            i = j
        # * Diphthong
        else:
            di = _di_to_vis[ph_list[i]]
            j = i + 1
            while j < len(ph_list) and ph_list[j] == ph_list[i]:
                j += 1
            k = int(math.ceil((i + j) / 2))
            for _ in range(i, k):
                ret.append(di[0])
            for _ in range(k, j):
                ret.append(di[1])
            i = j

    # assert len(ret) == len(ph_list)
    # for x, y in zip(ph_list, ret):
    #     if x == "HH" or x in _di_to_vis:
    #         print(x, y)
    return ret
