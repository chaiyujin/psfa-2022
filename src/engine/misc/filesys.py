import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ENGINE")


def ancestor(path: str, level: int = 1) -> str:
    assert level >= 1
    ret = path
    for _ in range(level):
        ret = os.path.dirname(ret)
    return ret


def find_files(directory: str, pattern: str, recursive: bool = True, abspath: bool = False) -> List[str]:
    if not os.path.exists(directory):
        return []

    file_list: List[str] = []
    regex = re.compile(pattern)
    for root, _, files in os.walk(directory):
        if os.path.basename(root) in [".AppleDouble", ".DS_Store"]:
            continue
        for f in files:
            if regex.match(f) is not None:
                file_list.append(os.path.join(root, f))
        if not recursive:
            break
    return [os.path.abspath(x) if abspath else os.path.relpath(x) for x in sorted(file_list)]


def find_dirs(directory: str, pattern: str, recursive: bool = True, abspath: bool = False) -> List[str]:
    if not os.path.exists(directory):
        return []

    dir_list: List[str] = []
    regex = re.compile(pattern)
    for root, subdirs, _ in os.walk(directory):
        if os.path.basename(root) in [".AppleDouble", ".DS_Store"]:
            continue
        for f in subdirs:
            if regex.match(f) is not None:
                dir_list.append(os.path.join(root, f))
        if not recursive:
            break
    return [os.path.abspath(x) if abspath else os.path.relpath(x) for x in sorted(dir_list)]


def maybe_in_dirs(
    filename: str,
    possible_roots: Optional[List[str]] = None,
    possible_exts: Optional[List[str]] = None,
    must_be_found: bool = False,
) -> str:
    """return first existing filepath according to filename, possible roots and extensions"""

    def _find_exts(path):
        if os.path.exists(path):
            return path
        if possible_exts is not None:
            assert isinstance(possible_exts, (list, tuple))
            for ext in possible_exts:
                if ext[0] != ".":
                    ext = "." + ext
                path = os.path.splitext(path)[0] + ext
                if os.path.exists(path):
                    return path
        return None

    assert (possible_roots is None) or isinstance(
        possible_roots, (list, tuple)
    ), "'possible_roots' should be list or tuple. not {}".format(possible_roots)
    assert (possible_exts is None) or isinstance(
        possible_exts, (list, tuple)
    ), "'possible_exts' should be list or tuple. not {}".format(possible_exts)

    fpath = _find_exts(filename)
    if fpath is None and possible_roots is not None:
        for root in possible_roots:
            if root is None or not os.path.exists(root):
                continue
            fpath = _find_exts(os.path.join(root, filename))
            if fpath is not None:
                # logger.debug("Find file at: {}".format(fpath))
                return fpath
    if must_be_found:
        assert fpath is not None, f"Failed to find file: {filename}"
    return fpath


def maybe_remove_end_separator(path: str) -> str:
    if len(path) > 0 and path[-1] in ["/", "\\"]:
        return path[:-1]
    else:
        return path
