import os
from typing import List, Optional

from src.datasets.utils_seq import Range, interp


def load_labels(config, frm_range: Range, a: float, data_dir: str, sub_path: str) -> Optional[List[str]]:
    src_path = os.path.join(data_dir, sub_path)
    if not os.path.exists(src_path):
        return None

    labels = []
    eof = False
    txt, xmin, xmax = "#", -1000, -1000
    with open(src_path) as fp:
        for i in frm_range:
            # * interploate 'a'
            idx = i + a
            # * get ts
            sec = idx / frm_range.fps
            while sec >= xmax:
                if not eof:
                    line = fp.readline().strip()
                    if len(line) == 0:
                        eof = True
                if eof:
                    txt, xmin, xmax = "#", 0, 100000
                else:
                    # ss = line.split(",")
                    # assert len(ss) == 3, "wrong line: {}, {}".format(line, len(ss))
                    txt, xmin, xmax = line.split(",")
                    xmin = float(xmin)
                    xmax = float(xmax)
            if sec < 0:
                txt = "#"
            else:
                assert xmin <= sec < xmax, "{}, {}, {}".format(xmin, sec, xmax)
            labels.append(txt)
    return labels
