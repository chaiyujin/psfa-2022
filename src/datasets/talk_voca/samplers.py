import enum
import re
from logging import getLogger
from os import wait
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.ma import count
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from ..flame_mesh.flame_mesh import FlameMeshDataset

log = getLogger("Sampler")


class SpeakerPairDataSampler(Sampler):
    def __init__(self, dataset: FlameMeshDataset, batch_size: int, shuffle: bool):
        super().__init__(dataset)
        assert batch_size % 2 == 0, f"[SpeakerPairDataSampler]: batch_size {batch_size} is not even"
        assert shuffle, "[SpeakerPairDataSampler]: shuffle should always be true!"
        self._batch_size = batch_size
        self._dataset = dataset
        # Collect speaker data list
        self._idx_of_spk: Dict[str, List[int]] = dict()
        for i_item, subseq in enumerate(dataset.sub_ranges):
            i_data = subseq.sidx  # the global sequence id for this sub-sequence
            spk = dataset.data_dicts[i_data].speaker  # the speaker
            assert spk in dataset.speakers, f"Unknown speaker '{spk}' found in dataset"
            if spk not in self._idx_of_spk:
                self._idx_of_spk[spk] = []
            self._idx_of_spk[spk].append(i_item)
        # Find a good size, so that in every epoch, speakers appear evenly.
        # Also, make sure each batch, a speaker has data of even number.
        self._n_spks = len(self._idx_of_spk)
        self._n_each_spk = min(len(vals) for _, vals in self._idx_of_spk.items())
        self._n_each_spk = self._n_each_spk // 2 * 2  # make it even
        self._n_batches = self._n_each_spk * self._n_spks // self._batch_size

        # init indices
        self.indices = np.arange(self._n_each_spk * self._n_spks)

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        # generate indices
        self.generate_indices()

        # generate batches
        for i in range(len(self)):
            s = i * self._batch_size
            e = i * self._batch_size + self._batch_size
            batch = list(self.indices[s:e])
            if len(batch) == 0:
                break
            yield batch

    def generate_indices(self):
        indices = []
        for _, vals in self._idx_of_spk.items():
            indices.append(np.random.choice(vals, self._n_each_spk, replace=False).astype(np.int64))
        self.indices = np.concatenate(indices)
        # shuffle, but adjacent data are kept together
        iidx = np.arange(len(self.indices) // 2)
        np.random.shuffle(iidx)
        iidx = np.repeat(np.expand_dims(iidx * 2, axis=1), repeats=2, axis=1)  # [i*2, i*2], [j*2, j*2], ...
        iidx[:, 1] += 1  # [i*2, i*2+1], [j*2, j*2+1], ...
        iidx = iidx.flatten()  # i*2, i*2+1, j*2, j*2+1, ...
        self.indices = self.indices[iidx]  # adjacent data, such as (0, 1) or (10, 11) are from same speaker.


class SharedTextSampler(Sampler):
    def __init__(
        self,
        dataset: FlameMeshDataset,
        batch_size: int,
        shuffle: bool,
        hop_paired: int = 1,
        pairs_from_same_speaker: bool = True,
        pairs_from_diff_speaker: bool = True,
    ):
        super().__init__(dataset)
        assert batch_size % 2 == 0, f"[SharedTextSampler]: batch_size {batch_size} is not even"
        assert shuffle, "[SharedTextSampler]: shuffle should always be true!"
        assert hasattr(dataset, "dtw_pairs"), "[SharedTextSampler]: dataset doesn't have 'dtw_pairs'!"
        self._batch_size = batch_size // 2
        self._dataset = dataset
        dtw_pairs = dataset.dtw_pairs
        real3d_speakers = set(dataset.real3d_speakers)

        log.info(
            "SharedTextSampler: same_speaker {}, diff_speaker {}".format(
                pairs_from_same_speaker, pairs_from_diff_speaker
            )
        )

        # Collect speaker data list
        self._items_of_tag: Dict[str, List[int]] = {}
        for i_item, subseq in enumerate(tqdm(dataset.sub_ranges)):
            spk = dataset.data_dicts[subseq.sidx].speaker  # the speaker
            sid = dataset.data_dicts[subseq.sidx].seq_id  # the seq_id
            tag = f"{spk}-{sid}"
            if spk not in dataset.speakers:
                continue
            if tag not in self._items_of_tag:
                self._items_of_tag[tag] = []
            self._items_of_tag[tag].append(i_item)

        # Collect pairs
        tags_paired_diff = set()
        self._item_pairs: List[Tuple[int, int]] = []
        if pairs_from_diff_speaker:
            hop = hop_paired
            for pair in dtw_pairs:
                tag0, tag1 = pair.split(",")
                # same speaker
                if tag0.split("-")[0] == tag1.split("-")[0]:
                    continue
                # not real3d speakers
                if (tag0.split("-")[0] not in real3d_speakers) or (tag1.split("-")[0] not in real3d_speakers):
                    continue
                # not paired
                if (tag0 not in self._items_of_tag) or (tag1 not in self._items_of_tag):
                    continue
                # paired sub-sequence
                vals0 = self._items_of_tag[tag0]
                vals1 = self._items_of_tag[tag1]
                count = 0
                for x in vals0[::hop]:
                    for y in vals1[::hop]:
                        x0, x1 = x, min(x + hop, vals0[-1] + 1)
                        y0, y1 = y, min(y + hop, vals1[-1] + 1)
                        self._item_pairs.append(((x0, x1), (y0, y1)))
                        count += 1
                tags_paired_diff.add(tag0)
                tags_paired_diff.add(tag1)
                # print(tag0, tag1, count)
        # same speaker pairs
        if pairs_from_same_speaker:
            hop = hop_paired
            for tag, vals in self._items_of_tag.items():
                # ignore those paired with different speakers
                if tag in tags_paired_diff:
                    continue
                count = 0
                for x in vals[::hop]:
                    for y in vals[::hop]:
                        if y <= x:
                            continue
                        x0, x1 = x, min(x + hop, vals[-1] + 1)
                        y0, y1 = y, min(y + hop, vals[-1] + 1)
                        self._item_pairs.append(((x0, x1), (y0, y1)))
                        count += 1
                # print(tag, count)

        # Find a good size, so that in every epoch, speakers appear evenly.
        # Also, make sure each batch, a speaker has data of even number.
        self._n_batches = len(self._item_pairs) // self._batch_size
        # print(self._n_batches)
        # quit(1)

        # init indices
        self.indices = np.arange(self._n_batches * self._batch_size)

    def __len__(self):
        # return 1
        return self._n_batches

    def __iter__(self):
        # generate indices
        self.generate_indices()

        # generate batches
        for i in range(len(self)):
            s = i * self._batch_size
            e = i * self._batch_size + self._batch_size
            batch = []
            for k in self.indices[s:e]:
                (xs, xe), (ys, ye) = self._item_pairs[k]
                x = np.random.randint(xs, xe)
                y = np.random.randint(ys, ye)
                batch.append(x)
                batch.append(y)
            if len(batch) == 0:
                break
            yield batch

    def generate_indices(self):
        self.indices = np.arange(self._n_batches * self._batch_size)
        np.random.shuffle(self.indices)


class BatchSamplerWithDtwPair(Sampler):
    def __init__(
        self,
        dataset: FlameMeshDataset,
        batch_size: int,
        shuffle: bool,
        pairs_from_same_speaker: bool = True,
        pairs_from_diff_speaker: bool = True,
    ):
        super().__init__(dataset)
        assert batch_size % 2 == 0, f"[SharedTextSampler]: batch_size {batch_size} is not even"
        assert shuffle, "[SharedTextSampler]: shuffle should always be true!"
        assert hasattr(dataset, "dtw_pairs"), "[SharedTextSampler]: dataset doesn't have 'dtw_pairs'!"
        self._full_bsz = batch_size
        self._dataset = dataset
        dtw_pairs = dataset.dtw_pairs
        real3d_speakers = set(dataset.real3d_speakers)

        log.info(
            "BatchSamplerWithDtwPair: same_speaker {}, diff_speaker {}".format(
                pairs_from_same_speaker, pairs_from_diff_speaker
            )
        )

        def is_celeb(spk):
            return re.match(r"(m|f)\d\d\d_.*", spk)

        # Collect speaker data list
        self._celeb: Optional[str] = None
        self._items_of_tag: Dict[str, List[int]] = {}
        self._items_of_spk: Dict[str, List[int]] = {}
        count_all_items = 0
        for i_item, subseq in enumerate(tqdm(dataset.sub_ranges)):
            spk = dataset.data_dicts[subseq.sidx].speaker  # the speaker
            sid = dataset.data_dicts[subseq.sidx].seq_id  # the seq_id
            tag = f"{spk}-{sid}"
            if spk not in dataset.speakers:
                continue
            if tag not in self._items_of_tag:
                self._items_of_tag[tag] = []
            self._items_of_tag[tag].append(i_item)
            if spk not in self._items_of_spk:
                self._items_of_spk[spk] = []
            self._items_of_spk[spk].append(i_item)
            count_all_items += 1
            # set the celeb
            if is_celeb(spk):
                assert self._celeb is None or self._celeb == spk
                self._celeb = spk

        # Collect pairs
        self._items_paired_with_tag = dict()
        self._items: List[Tuple[int, str, str]] = []
        for tag, items in self._items_of_tag.items():
            self._items_paired_with_tag[tag] = []
            spk = tag.split("-")[0]

            def _extend(tag_paired):
                for x in self._items_of_tag[tag_paired]:
                    self._items_paired_with_tag[tag].append(x)

            def _extend_spk(spk_paired):
                assert spk_paired == tag.split("-")[0]
                for x in self._items_of_spk[spk_paired]:
                    self._items_paired_with_tag[tag].append(x)

            # find paired tag
            if pairs_from_diff_speaker:
                for pair in dtw_pairs:
                    tag0, tag1 = pair.split(",")
                    if tag0.split("-")[0] == tag1.split("-")[0]:  # skip same speaker pair
                        continue
                    if (tag0.split("-")[0] not in real3d_speakers) or (tag1.split("-")[0] not in real3d_speakers):
                        continue
                    if (tag0 not in self._items_of_tag) or (tag1 not in self._items_of_tag):  # no data
                        continue
                    if tag0 == tag:
                        _extend(tag1)
                    elif tag1 == tag:
                        _extend(tag0)
                    # print("pair tag", tag0, tag1)

            n_pairs_diff_spk = len(self._items_paired_with_tag[tag])

            # pair with self only if no other aligned data
            if pairs_from_same_speaker:
                if len(self._items_paired_with_tag[tag]) == 0:
                    if is_celeb(spk):
                        log.info("use pairs from same speaker: {}".format(spk))
                        _extend_spk(spk)
                    else:
                        assert spk.startswith("FaceTalk_")
                        # log.info("use pairs from same sequence: {}".format(tag))
                        _extend(tag)

            n_pairs_same_spk = len(self._items_paired_with_tag[tag]) - n_pairs_diff_spk

            if is_celeb(spk):
                log.info(f"{tag} has {n_pairs_diff_spk} pairs from diff, and {n_pairs_same_spk} pairs from same.")

            for x in items:
                self._items.append((x, tag, spk))

        # sanity check
        assert len(self._items) == count_all_items
        assert len(self._items) == sum(len(v) for _, v in self._items_of_spk.items())
        # different modes
        self._need_pair = pairs_from_diff_speaker or pairs_from_same_speaker
        self._batch_size = (self._full_bsz // 2) if self._need_pair else self._full_bsz
        # self._n_batches = len(self._items) // self._batch_size
        self._n_batches = len(self._items) // self._full_bsz
        self.__generate_random_indices()

    def __generate_random_indices(self):
        if not self._need_pair:
            self.indices = np.arange(len(self._items))
            np.random.shuffle(self.indices)
            # drop last
            cnt = self._n_batches * self._batch_size
            self.indices = self.indices[:cnt]
            # logging
            log.info("Sampled data indices: {} ...".format(self.indices[:10]))
        else:
            # first, we sample data for each speaker
            indices = []
            items_used = {k: set() for k in self._items_of_spk}
            for i, (x, _, spk) in enumerate(self._items):
                if len(items_used[spk]) * 2 >= len(self._items_of_spk[spk]):
                    continue
                indices.append(i)
                items_used[spk].add(x)
            for spk in items_used:
                print("Sample {}(out of {}) for {}".format(len(items_used[spk]), len(self._items_of_spk[spk]), spk))
            self.indices = np.asarray(indices, dtype=np.int64)
            np.random.shuffle(self.indices)
            # drop last
            cnt = self._n_batches * self._batch_size
            assert len(self.indices) >= cnt
            self.indices = self.indices[:cnt]
            # logging
            log.info("Sampled data indices: {} ...".format(self.indices[:10]))

            # for celeb
            self._random_items_of_spk = dict()
            self._random_items_idx = dict()
            if self._celeb is not None:
                items = np.array(
                    [x for x in self._items_of_spk[self._celeb] if x not in items_used[self._celeb]], copy=True
                )
                np.random.shuffle(items)
                if len(items) < len(items_used[self._celeb]):
                    extra_items = np.array(self._items_of_spk[self._celeb], copy=True)
                    np.random.shuffle(extra_items)
                    extra_items = extra_items[: len(items_used[self._celeb]) - len(items)]
                    items = np.concatenate((items, extra_items))
                assert len(items) >= len(items_used[self._celeb])
                self._random_items_of_spk[self._celeb] = items
                self._random_items_idx[self._celeb] = 0
                log.info("Paired items items for celeb: {} ...".format(self._random_items_of_spk[self._celeb][:10]))

    def __get_paired(self, tag, spk):
        if spk != self._celeb:
            y = np.random.choice(self._items_paired_with_tag[tag])
        else:
            y = self._random_items_of_spk[spk][self._random_items_idx[spk]]
            self._random_items_idx[spk] += 1
        return y

    def __len__(self):
        # return 1
        return self._n_batches

    def __iter__(self):
        # generate indices
        self.__generate_random_indices()

        # generate batches
        for i in range(len(self)):
            s = i * self._batch_size
            e = i * self._batch_size + self._batch_size
            batch = []
            for k in self.indices[s:e]:
                x, tag, spk = self._items[k]
                batch.append(x)
                if self._need_pair:
                    # y = np.random.choice(self._items_paired_with_tag[tag])
                    y = self.__get_paired(tag, spk)
                    batch.append(y)
            if len(batch) == 0:
                break
            assert len(batch) == self._full_bsz
            yield batch
