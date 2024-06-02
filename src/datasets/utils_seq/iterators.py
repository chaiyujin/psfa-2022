import logging
from typing import List, Tuple, Union

import numpy as np
import torch
from omegaconf import ListConfig

from .types import Range, SeqCoord

# log = logging.getLogger(__name__)


class SequenceIterator(object):
    def __init__(self, dataset, seq_list: Union[str, List[int], Tuple[int, ...]], hop_subseq: int):

        # Team object reference
        self._dataset = dataset
        self._wanna_fps = dataset.wanna_fps

        # get source list
        if isinstance(seq_list, str):
            if seq_list == "random-one":
                seq_list = [np.random.randint(len(dataset.seq_ranges))]
            elif seq_list == "all":
                seq_list = list(range(len(dataset.seq_ranges)))
            elif seq_list == "one-per-speaker":
                seq_list = []
                visited = set()
                for i in range(len(dataset.seq_ranges)):
                    i_seq = dataset.seq_ranges[i].sidx
                    speaker = dataset.data_dicts[i_seq]["speaker:str"]
                    if speaker not in visited:
                        visited.add(speaker)
                        seq_list.append(i)
            else:
                raise ValueError("Unknown seq_list: {}".format(seq_list))

        assert isinstance(seq_list, (list, tuple, ListConfig)) and all(
            isinstance(x, int) for x in seq_list
        ), "'seq_list' must be str ('random-one', 'all'), List[int] or Tuple[int, ...]"
        self._seq_list = tuple(seq_list)
        self._hop = hop_subseq

        # init index
        self._index = 0
        # log.info("Generate videos of dataset for sequence: {}".format(self._seq_list))

    def __next__(self):
        if self._index < len(self._seq_list):
            full_seq: SeqCoord = self._dataset.seq_ranges[self._seq_list[self._index]]
            data_dict = self._dataset.data_dicts[full_seq.sidx]
            max_frames = data_dict.n_frames
            # split into sub-sequences
            subseqs: List[Tuple[int, Range]] = []
            hop_frms = int(self._hop * full_seq.fps / self._wanna_fps)
            seq_frms = int(self._dataset.seq_duration * full_seq.fps)
            for i in range(0, max_frames, hop_frms):
                idx = i
                jdx = i + (seq_frms + 1)  # one more frame for interplation
                if jdx >= max_frames:
                    jdx = max_frames - 1
                    idx = jdx - (seq_frms + 1)
                subseqs.append((full_seq.sidx, Range(idx, seq_frms, full_seq.fps)))
            self._index += 1
            return full_seq.tag, subseqs, data_dict
        # End of Iteration
        raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._seq_list)


def _trim_data_and_get_batch(_data, _stt_frm, shared_keys=[]):
    batch = dict()
    for key in _data:
        val = _data[key]
        if isinstance(val, dict):
            batch[key] = _trim_data_and_get_batch(_data[key], _stt_frm, shared_keys)
        elif isinstance(val, int):
            batch[key] = torch.LongTensor([val])
        elif isinstance(val, float):
            batch[key] = torch.FloatTensor([val])
        elif key in shared_keys:
            batch[key] = _data[key].unsqueeze(0)
        elif torch.is_tensor(val):
            _data[key] = _data[key][_stt_frm:]
            batch[key] = _data[key].unsqueeze(0)
        elif isinstance(val, str):
            batch[key] = [val]
        elif key == "style_comb":
            batch[key] = torch.tensor(_data[key]).unsqueeze(0)
        elif isinstance(_data[key], Range) or key in ["load_kwargs", "tgt_frm_range"]:
            continue
        else:
            _data[key] = _data[key][_stt_frm:]
            batch[key] = _data[key]
        # to cuda
        if torch.cuda.is_available():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].cuda()
    return batch


class SubSeqIterator(object):
    def __init__(self, dataset, subseqs: List[Tuple[int, Range]], max_frames=None, **dkwargs):
        self._dataset = dataset
        self._subseqs = subseqs
        self._dkwargs = dkwargs
        self._max_frames = max_frames
        self._cnt_frames = 0
        self._index = 0

    def __next__(self):
        if self._index < len(self._subseqs) and (self._max_frames is None or self._cnt_frames < self._max_frames):
            # get current idx and move to next
            sidx, frm_range = self._subseqs[self._index]
            subseq_data = self._dataset.get_data(sidx, frm_range, a=0, is_generating=True)
            self._index += 1

            # overwrite data with dkwargs
            for key, val in self._dkwargs.items():
                # print(f"Overwrite data: {key}, {val}")
                subseq_data[key] = val

            # # get frames
            # subseq_n_frames = -1
            # for key in ["image", "offsets_tracked", "offsets_REAL"]:
            #     if subseq_data.get(key) is not None:
            #         subseq_n_frames = subseq_data[key].shape[0]
            #     if subseq_n_frames >= 0:
            #         break
            # assert subseq_n_frames >= 0, "Failed to get sub-sequence n_frames"

            tgt_range = frm_range.convert_to(self._dataset.wanna_fps)

            subseq_frameid_list = list(iter(tgt_range))
            # assert len(subseq_frameid_list) == subseq_n_frames, "len is {}, n-frame is {}".format(
            #     len(subseq_frameid_list), subseq_n_frames
            # )
            # check max frames
            # if self._max_frames is not None and self._cnt_frames + len(subseq_frameid_list) > self._max_frames:
            #     subseq_frameid_list = subseq_frameid_list[: self._max_frames - self._cnt_frames]

            # update count frames
            self._cnt_frames += len(subseq_frameid_list)

            # get batch
            if len(subseq_frameid_list) == 0:
                batch = None
            else:
                batch = _trim_data_and_get_batch(subseq_data, 0, shared_keys=["idle_verts"])

            # return
            return subseq_frameid_list, batch

        raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return len(self._subseqs)
