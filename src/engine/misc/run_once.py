import functools
import sys
from inspect import currentframe, getframeinfo
from typing import Dict


class run_once(object):
    __has_run__: Dict[str, bool] = dict()

    def __init__(self, tag=""):
        frameinfo = getframeinfo(currentframe().f_back)
        self.key = "[{}]{}:{}".format(tag, frameinfo.filename, frameinfo.lineno)

    def __enter__(self):
        if run_once.__has_run__.get(self.key) is not None:
            # https://code.google.com/archive/p/ouspg/wikis/AnonymousBlocksInPython.wiki
            sys.settrace(lambda *args, **keys: None)
            frame = currentframe().f_back
            frame.f_trace = self.trace
        else:
            run_once.__has_run__[self.key] = True

    def trace(self, frame, event, arg):
        raise Exception()

    def __exit__(self, *args):
        return True

    def __call__(self, method):
        @functools.wraps(method)
        def wrapper(*args, **kw):
            if run_once.__has_run__.get(self.key) is None:
                run_once.__has_run__[self.key] = True
                wrapper.results = method(*args, **kw)
            return wrapper.results

        wrapper.results = None
        return wrapper
