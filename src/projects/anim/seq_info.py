from typing import Optional, Tuple


class SeqInfo(object):
    def __init__(
        self,
        n_frames,
        padding: Tuple[int, int] = (0, 0),
        fps: Optional[float] = None,
        speaker: Optional[str] = None,
        tag: Optional[str] = None,
    ):
        assert len(padding) == 2
        self._n_frames = n_frames
        self._padding = (int(padding[0]), int(padding[1]))
        # Optional
        self._fps = fps
        self._speaker = speaker
        self._tag = tag

    @property
    def n_frames_valid(self):
        return self._n_frames

    @property
    def n_frames_padded(self):
        return self._n_frames + sum(self._padding)

    @property
    def has_padding(self) -> bool:
        return sum(self._padding) > 0

    @property
    def padding(self) -> Tuple[int, int]:
        return self._padding

    @property
    def fps(self) -> float:
        assert self._fps is not None, "You didn't set fps for this SeqInfo!"
        return self._fps

    @property
    def speaker(self) -> str:
        assert self._speaker is not None, "You dint's set speaker for this SeqInfo!"
        return self._speaker

    @property
    def tag(self) -> str:
        assert self._tag is not None, "You dint's set tag for this SeqInfo!"
        return self._tag

    def remove_padding(self, tensor):
        if tensor is None:
            return tensor

        # check length
        assert tensor.ndim >= 2
        assert tensor.shape[1] == self.n_frames_padded, "given {}, but wanna {} length".format(
            tensor.shape, self.n_frames_padded
        )
        # remove padding
        if sum(self._padding) > 0:
            pl, pr = self._padding
            tensor = tensor[:, pl : tensor.shape[1] - pr]

        assert tensor.shape[1] == self.n_frames_valid
        return tensor
