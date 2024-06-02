from typing import Any, Optional


class dotdict(dict):
    def __getitem__(self, key):
        idx = key.find(".")
        if idx < 0:
            return super().__getitem__(key)
        return dotdict.fetch(self, key)

    def __setitem__(self, key, val):
        idx = key.find(".")
        if idx < 0:
            super().__setitem__(key, val)
            return
        dotdict.put(self, key, val)

    def __getattr__(self, k) -> Any:
        if k in self.__dict__:
            return super().__getattr__(k)
        else:
            return super().__getitem__(k)

    def __setattr__(self, k, v):
        if k in self.__dict__:
            super().__setattr__(k, v)
        else:
            super().__setitem__(k, v)

    @staticmethod
    def fetch(data, key):
        idx = key.find(".")
        if idx < 0:
            return data.__getitem__(key)
        # nested
        full_key = key
        k, key = key[:idx], key[idx + 1 :]
        try:
            sub_dict = data.__getitem__(k)
            if not isinstance(sub_dict, dict):
                return None
                # raise KeyError(f"cannot get '{key}' from {type(sub_dict)}")
            else:
                return dotdict.fetch(sub_dict, key)
        except KeyError:
            # raise KeyError(f"cannot get '{full_key}'")
            return None

    @staticmethod
    def put(data, key, val):
        idx = key.find(".")
        if idx < 0:
            data.__setitem__(key, val)
            return
        # nested
        full_key = key
        k, key = key[:idx], key[idx + 1 :]
        sub_dict = data.__getitem__(k)
        try:
            if not isinstance(sub_dict, dict):
                raise KeyError(f"cannot put '{key}' into {type(sub_dict)}")
            else:
                dotdict.put(sub_dict, key, val)
        except KeyError:
            raise KeyError(f"cannot put '{full_key}'")


if __name__ == "__main__":
    d = dotdict(a=0, b=100, xyz=dict(x=1, y=2, z=3))
    print(d.get("a"))
    print(dotdict.fetch(d, "a"))
    print(dotdict.fetch(d, "b"))
    print(dotdict.fetch(d, "xyz.x"))
    dotdict.put(d, "xyz.z", -10)
    d["xyz.x"] = -100
    print(d["xyz.z"], d["xyz.x"])
    print(d)
