import logging
import os
import re
from pydoc import locate
from typing import Any, Dict, List, Tuple, Type, Union

import pandas

# from tqdm.rich import tqdm

logger = logging.getLogger("ENGINE")


_types: List[str] = ["int", "str", "path", "float"]


class Metadata(str):
    def __init__(self, *args, **kwargs):
        super().__init__()
        splited = self.split(":")
        assert len(splited) == 2, "meta `{}` is not in format <name>:<type>".format(self)
        assert splited[1] in _types, "<type> should be in {}".format(_types)
        type_str = splited[1]
        self._is_path = type_str == "path"
        if self._is_path:
            type_str = "str"
        self._name: str = splited[0]
        self._type: Type = locate(type_str)

    @property
    def name(self):
        return self._name

    @property
    def value_type(self):
        return self._type

    @property
    def is_path(self):
        return self._is_path


class DataDict(dict):
    __regex__ = re.compile("^__.*__$")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__metadata__: List[Metadata] = [Metadata(x) for x in self.keys()]
        self.__key_mapping__: Dict[str, Tuple[str, Type]] = {x.name: (x, x.value_type) for x in self.__metadata__}

    @property
    def metadata(self) -> List[Metadata]:
        return self.__metadata__

    def get(self, k):
        try:
            if k.find(":") > 0:
                key, value_type = k, Metadata(k).value_type
            else:
                key, value_type = self.__key_mapping__[k]
        except Exception:
            raise KeyError(f"'{k}' is not valid key!")
        return value_type(super().__getitem__(key))

    def set(self, k, v):
        if k.find(":") > 0:
            if k not in self:
                raise KeyError(f"'{k}' is not valid key!")
            key, value_type = k, Metadata(k).value_type
        else:
            if k not in self.__key_mapping__:
                raise KeyError(f"'{k}' is not valid key!")
            key, value_type = self.__key_mapping__[k]
        if not isinstance(v, value_type):
            raise TypeError(f"Given value for key '{k}' should be '{value_type}', but '{type(v)}' is found!")
        # set
        assert key in self, "Impossible!"
        super().__setitem__(key, v)

    def __getitem__(self, k) -> Any:
        return self.get(k)

    def __setitem__(self, k, v):
        return self.set(k, v)

    def __getattr__(self, k) -> Any:
        if k in self.__dict__:
            super().__getattr__(k)
        else:
            return self.get(k)

    def __setattr__(self, k, v):
        if DataDict.__regex__.match(k):
            super().__setattr__(k, v)
        else:
            self.set(k, v)

    def to_dict(self):
        return {k.split(":")[0]: self.get(k) for k in self.__metadata__}


def _get_types(metadata: List[str]) -> List[object]:
    types = []
    for meta in metadata:
        name, type_str = meta.split(":")
        if type_str == "path":
            type_str = "str"
        t = locate(type_str)
        types.append(t)
    return types


def write_csv(
    output_file: str,
    metadata: List[Metadata],
    data_dicts: List[DataDict],
    spliter: str = ",",
    use_relpath: bool = True,
):
    # ignore the empty dicts
    if len(data_dicts) == 0:
        return

    # check metadata: pass

    # fix extension to '.csv'
    output_file = os.path.splitext(output_file)[0] + ".csv"
    # write data_dicts
    dirname = os.path.dirname(output_file)
    with open(output_file, "w", encoding="utf-8") as fp:
        fp.write(spliter.join(metadata) + "\n")
        for data in data_dicts:
            line = spliter.join(
                [
                    os.path.relpath(str(data[meta]), dirname) if (meta.is_path and use_relpath) else str(data[meta])
                    for meta in metadata
                ]
            )
            fp.write(line + "\n")


def read_csv(
    csv_path: str,
    spliter: str = ",",
    is_relpath: bool = True,
) -> Tuple[List[Metadata], List[DataDict]]:
    assert os.path.exists(csv_path), "Failed to find csv file: '{}'".format(csv_path)
    dirname: str = os.path.dirname(csv_path)
    data_dicts: List[DataDict] = []

    df = pandas.read_csv(csv_path, sep=spliter)
    # check metadata
    metadata: List[Metadata] = [Metadata(x) for x in df.columns.values]

    # read tuples
    # for row in tqdm(df.values, desc="read csv", total=len(df.values)):
    for row in df.values:
        data = DataDict(
            {
                meta: (os.path.join(dirname, str(d)) if (meta.is_path and is_relpath) else meta.value_type(d))
                for d, meta in zip(row, metadata)
            }
        )
        data_dicts.append(data)

    return metadata, data_dicts


def read_csv_or_list(
    sources: Union[str, List[str]],
    spliter: str = ",",
    same_metadata: bool = True,
    is_relpath: bool = True,
) -> Tuple[List[Metadata], List[DataDict]]:

    cur_meta: List[Metadata] = []
    cur_dicts: List[DataDict] = []

    csv_list: List[str] = [sources] if isinstance(sources, str) else sources
    assert isinstance(csv_list, list)

    # read each csv file
    for csv_file in csv_list:
        assert isinstance(csv_file, str)
        metadata, data_dicts = read_csv(csv_file, spliter, is_relpath=is_relpath)
        metadata = sorted(metadata)
        # check metadata
        if len(cur_meta) == 0:
            cur_meta = metadata
        else:
            # check same
            if same_metadata:
                assert (
                    cur_meta == metadata
                ), "[read_csv_list]: given csv files have different metadata, but same_meta=True."
            # merge metadata
            else:
                cur_meta = sorted(list(set(cur_meta) | set(metadata)))
        cur_dicts.extend(data_dicts)
    # fill missing data
    if not same_metadata:
        for data in cur_dicts:
            for meta in cur_meta:
                if meta not in data:
                    super(DataDict, data).__setitem__(meta, None)
    return cur_meta, cur_dicts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    args = parser.parse_args()
