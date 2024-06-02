from typing import Dict, List, Optional

import numpy as np
from torch.utils.data.sampler import Sampler

from src.engine.logging import get_logger

from ..utils_seq.types import SeqCoord
from .anim import AnimBaseDataset

log = get_logger("Sampler")


class LitmitedFramesBatchSampler(Sampler):
    def __init__(self, dataset: AnimBaseDataset, max_frames: int, batch_size: int, shuffle: bool):
        super().__init__(dataset)

        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._indices_of_speaker: Dict[str, List[int]] = {key: [] for key in dataset.speakers}

        # find items for each clip source
        for i_item, subseq in enumerate(dataset.sub_ranges):
            subseq: SeqCoord
            i_data = subseq.sidx
            key = dataset.data_dicts[i_data].speaker
            assert key in self._indices_of_speaker, "Unknown speaker: {}".format(key)
            self._indices_of_speaker[key].append(i_item)

        total_vals, to_remove = [], []
        for key, vals in self._indices_of_speaker.items():
            assert len(vals) == len(set(vals))
            if len(vals) == 0:
                to_remove.append(key)
            else:
                total_vals += vals
                log.info("- {} subseqs for speaker {}".format(len(vals), key))
        # remove item with nothing
        for key in to_remove:
            self._indices_of_speaker.pop(key)
        assert len(total_vals) == len(set(total_vals))

        # find a good size
        self._n_speakers = len(self._indices_of_speaker.keys())
        self._item_length = min(len(vals) for _, vals in self._indices_of_speaker.items())
        self._item_length = min(self._item_length, max_frames) // self._n_speakers
        self._length = self._item_length * len(self._indices_of_speaker) // self._batch_size
        log.info(
            "[limited frames sampler]: with {} data per speaker ({}) for each epoch, totoally {} batches".format(
                self._item_length, self._n_speakers, self._length
            )
        )

        # init random indices
        self.generate_indices()

    def __len__(self):
        # return 1
        return self._length

    def generate_indices(self):
        indices = []
        for _, vals in self._indices_of_speaker.items():
            indices += list(np.random.choice(vals, self._item_length, replace=False))
        indices = np.asarray(indices, dtype=np.int64)

        if self._shuffle:
            np.random.shuffle(indices)

        self.indices = indices

    def __iter__(self):
        # random from all data
        self.generate_indices()

        # log.warning('random indices once', indices)
        # generate batches
        for i in range(len(self)):
            s = i * self._batch_size
            e = i * self._batch_size + self._batch_size
            batch = list(self.indices[s:e])
            if len(batch) == 0:
                raise StopIteration
            yield batch
