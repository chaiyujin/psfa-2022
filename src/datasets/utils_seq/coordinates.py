from typing import List, Tuple

from .types import FrmCoord, Range, SeqCoord


def process_data_dicts(
    data_dicts, speakers, used_seq_ids, seq_ids_used_duration, seq_duration: float, hop_duration: float
):
    # get all sub-sequence coordinate
    frm_coords: List[FrmCoord] = []  # Frame coordinate
    seq_ranges: List[SeqCoord] = []  # Sequences' frame ranges
    sub_ranges: List[SeqCoord] = []  # Sub-Seqs' frame ranges

    # * Find all coordinates and sub-sequences
    # ! WARNING: the coordinates is under original fps
    cur = 0
    new_data_dicts = []
    for data_dict in data_dicts:
        # * Filter speakers
        if speakers != "all" and data_dict.speaker not in speakers:
            continue
        # * Filter seq id
        if used_seq_ids is not None and data_dict.seq_id not in used_seq_ids:
            continue

        tag = "{}/{}".format(data_dict.speaker, data_dict.seq_id)
        new_data_dicts.append(data_dict)
        i_seq = len(new_data_dicts) - 1

        # * Amount of used
        n_frames = data_dict.n_frames

        def _used_frames(duration):
            if isinstance(duration, str) and duration.lower() in ["", "full", "all", "entire"]:
                used_frames = n_frames
            else:
                used_frames = int(float(duration) * data_dict.fps)
            return used_frames

        key_id = data_dict.seq_id.replace("-", "_")
        # all cases
        if "all" in seq_ids_used_duration:
            duration = seq_ids_used_duration["all"]
            n_frames = min(n_frames, _used_frames(duration))
        # specially set
        if key_id in seq_ids_used_duration:
            duration = seq_ids_used_duration[key_id]
            n_frames = min(n_frames, _used_frames(duration))

        # * Append into coords
        gidx_stt = cur
        for i_frm in range(n_frames):
            ts = float(i_frm) / data_dict.fps
            frm_coords.append(FrmCoord(gidx=cur, sidx=i_seq, fidx=i_frm, ts=ts))
            cur += 1
        gidx_end = gidx_stt + n_frames
        seq_ranges.append(SeqCoord(tag=tag, gidx_stt=gidx_stt, n_frames=n_frames, sidx=i_seq, fps=data_dict.fps))

        # * Segment into subsequces with overlap under original fps
        seq_frms = int(seq_duration * data_dict.fps)
        hop_frms = max(int(hop_duration * data_dict.fps), 1)
        for i in range(gidx_stt, gidx_end, hop_frms):
            idx = i
            jdx = i + (seq_frms + 1)  # one extra frame for interpolation
            if jdx >= gidx_end:
                jdx = gidx_end - 1
                idx = jdx - (seq_frms + 1)
            # assert idx >= stt_g_idx
            assert jdx < gidx_end
            sub_ranges.append(SeqCoord(tag="", gidx_stt=idx, n_frames=seq_frms, sidx=i_seq, fps=data_dict.fps))

    return new_data_dicts, frm_coords, seq_ranges, sub_ranges


def get_frm_range_from_sub_range(item, frm_coords) -> Tuple[int, Range]:
    sidx = item.sidx
    # global index
    gidx_stt = item.gidx_stt
    gidx_end = item.gidx_stt + item.n_frames + 1  # one extra frame for interpolation
    while frm_coords[gidx_stt].sidx != sidx:
        gidx_stt += 1
    assert frm_coords[gidx_stt].sidx == sidx
    assert frm_coords[gidx_end].sidx == sidx
    # internal frame index of i_seq
    stt_idx = frm_coords[gidx_stt].fidx
    # * end_idx may out of range
    # end_idx = frm_coords[gidx_end].fidx  # just for checking
    # assert stt_idx + item.n_frames + 1 == end_idx
    return sidx, Range(stt_idx, item.n_frames, item.fps)
