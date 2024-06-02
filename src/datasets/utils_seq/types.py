"""
* The frame is stored in :code:`class FrameCoord`.
    - gidx: the global index in all frames.
    - sidx: which sequence this frame belongs to.
    - fidx: the index in the sequence.
    - ts: the timestamp in the sequence
* The sub-sequence is stored in :code:`SubSeqCoord`.
    - fps: frame per second
    - gidx_stt: the global index of start frame.
    - gidx_end: the global index of end frame.
    - sidx: which sequence this sub-sequence belongs to.
    - tag: the tag of this sequence (sub-sequence)
"""


class FrmCoord:
    def __init__(self, gidx, sidx, fidx, ts):
        self.gidx = gidx
        self.sidx = sidx
        self.fidx = fidx
        self.ts = ts
        assert isinstance(self.gidx, int)
        assert isinstance(self.sidx, int)
        assert isinstance(self.fidx, int)
        assert isinstance(self.ts, (float, int))


class SeqCoord:
    def __init__(self, tag, gidx_stt, n_frames, sidx, fps):
        self.tag = tag
        self.gidx_stt = gidx_stt
        self.n_frames = n_frames
        self.sidx = sidx
        self.fps = fps
        assert isinstance(self.tag, str)
        assert isinstance(self.gidx_stt, int)
        assert isinstance(self.n_frames, int)
        assert isinstance(self.sidx, int)
        assert isinstance(self.fps, (float, int))


class Range:
    def __init__(self, stt_idx, n_frames, fps):
        self.stt_idx = stt_idx
        self.n_frames = n_frames
        self.fps = fps
        assert isinstance(self.stt_idx, int)
        assert isinstance(self.n_frames, int)
        assert isinstance(self.fps, (float, int))

    def convert_to(self, tar_fps):
        stt_idx = int(self.stt_idx * tar_fps / self.fps)
        n_frames = int(self.n_frames * tar_fps / self.fps)
        return Range(stt_idx, n_frames, tar_fps)

    def copy(self):
        return Range(self.stt_idx, self.n_frames, self.fps)

    def __iter__(self):
        return iter(range(self.stt_idx, self.stt_idx + self.n_frames))

    def __str__(self) -> str:
        return f"{self.stt_idx} ~ {self.stt_idx + self.n_frames - 1} ({self.fps})"
