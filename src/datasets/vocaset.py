import re

SPEAKERS = (
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170731_00024_TA",
    "FaceTalk_170809_00138_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170811_03275_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170908_03277_TA",
    "FaceTalk_170912_03278_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170915_00223_TA",
)
TRAIN_SPEAKERS = (
    "FaceTalk_170725_00137_TA",
    "FaceTalk_170728_03272_TA",
    "FaceTalk_170811_03274_TA",
    "FaceTalk_170904_00128_TA",
    "FaceTalk_170904_03276_TA",
    "FaceTalk_170912_03278_TA",
    "FaceTalk_170913_03279_TA",
    "FaceTalk_170915_00223_TA",
)
VALID_SPEAKERS = ("FaceTalk_170811_03275_TA", "FaceTalk_170908_03277_TA")
TEST_SPEAKERS = ("FaceTalk_170731_00024_TA", "FaceTalk_170809_00138_TA")


def speaker_to_char(speaker):
    assert speaker in SPEAKERS, "unknown speaker: {}".format(speaker)
    idx = SPEAKERS.index(speaker)
    return f"{idx:X}"


def char_to_speaker(c):
    idx = int(c, base=16)
    return SPEAKERS[idx]


def string_of_ignored_speakers(ignore_speakers):
    speakers = []
    for spk in SPEAKERS:
        if spk not in ignore_speakers:
            speakers.append(spk)
    return speakers_to_string(speakers)


def speakers_to_string(speakers, prefix="vocaspk"):
    # check
    speakers = set(speakers)
    for spk in speakers:
        if spk not in SPEAKERS:
            return "vocaukw"
    # to string
    binary = ""
    for spk in SPEAKERS:
        binary = ("1" if spk in speakers else "0") + binary
    binary = int(binary, base=2)
    postfix = f"{binary:03x}"
    string = f"{prefix}{postfix}"

    check_speakers = string_to_speakers(string, prefix)
    for spk in check_speakers:
        assert spk in speakers, "[vocaset::speakers_to_string] Check failed!"

    return string


def string_to_speakers(string, prefix="vocaspk"):
    match = re.match(prefix + r"([0-9a-fA-F]{3})", string)
    assert match is not None, "[vocaset::string_to_speakers] Wrong format string: {}".format(string)
    onehot = int(match.group(1), base=16)
    speakers = []
    for i, spk in enumerate(SPEAKERS):
        if onehot % 2 == 1:
            speakers.append(spk)
        onehot //= 2
    return speakers
