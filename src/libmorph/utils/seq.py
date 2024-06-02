def fold(x):
    if x is None:
        return None, None
    return x.contiguous().view(-1, *x.shape[2:]), x.shape[1]


def unfold(x, frm):
    if x is None or frm is None:
        return None
    if x.shape[0] == 1:
        return x.unsqueeze(1)
    return x.contiguous().view(x.shape[0] // frm, frm, *x.shape[1:])


def unfold_dict(data, frm):
    if frm is None:
        return
    for k in data:
        if isinstance(data[k], dict):
            unfold_dict(data[k], frm)
        else:
            data[k] = unfold(data[k], frm)
    return data


def fold_codes(code_dict):
    frm = None
    for key, val in code_dict.items():
        if val is None:
            continue
        # * check ndim
        ndim = 4 if key == "sh9" else 3
        if key not in ["shape", "tex"]:
            assert val.ndim == ndim or val.ndim == ndim - 1
            if val.ndim == ndim:
                if frm is not None:
                    assert frm == val.shape[1]
                frm = val.shape[1]

    if frm is None:
        return code_dict, None

    # * return new dict if fold
    ret = dict()
    for key, val in code_dict.items():
        if val is None:
            continue
        ret[key], _ = fold(val)
    return ret, frm
