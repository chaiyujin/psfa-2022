import os
import pickle
from typing import Any

import numpy as np


def to_np(array: Any, dtype: Any = np.float32) -> Any:
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)  # type: ignore


def chumpy_to_numpy(model_path: str):
    model_path = os.path.expanduser(model_path)
    with open(model_path, "rb") as f:
        flame_model = pickle.load(f, encoding="latin1")
    
    new_model = {}
    new_model["f"] = to_np(flame_model["f"], dtype="int32")
    new_model["v_template"] = to_np(flame_model["v_template"])
    new_model["shapedirs"] = to_np(flame_model["shapedirs"])
    new_model["posedirs"] = to_np(flame_model["posedirs"])
    new_model["J_regressor"] = to_np(flame_model["J_regressor"])
    new_model["weights"] = to_np(flame_model["weights"])
    new_model["kintree_table"] = to_np(flame_model["kintree_table"], dtype="int32")
    
    with open(os.path.splitext(model_path)[0] + "-np.pkl", "wb") as fp:
        pickle.dump(new_model, fp)


if __name__ == "__main__":
    _DIR = os.path.dirname(os.path.abspath(__file__))
    chumpy_to_numpy(os.path.join(_DIR, "generic_model.pkl"))
