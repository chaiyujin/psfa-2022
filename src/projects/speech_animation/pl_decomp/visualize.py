import numpy as np

from assets import get_vocaset_template_vertices
from src.engine.mesh_renderer import render, render_heatmap
from src.engine.painter import Text, draw_canvas, put_texts
from src.projects.anim.vis import register_painter


@register_painter
def demo_recons(cls, hparams, batch, results, bi, fi, A=256):

    # print_dict("batch", batch)
    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    y = _h("phase_1st.y_ip_rec_ani")
    Y = _h("Y_IP")
    if Y is None:
        return None

    # copy the non-face part of REAL data
    nidx = results["NON_FACE_VIDX"]
    # y[nidx] = Y[nidx]

    titles, rows, txts = [], [], []
    titles.append(("Recons", cls.colors.white))
    titles.append(("REAL", cls.colors.green))
    titles.append(("Error", cls.colors.red))

    # fmt: off
    # * Row 1
    row, txt = [], []
    dist_cm = np.linalg.norm(y - Y, axis=-1) * 100
    dist_cm[nidx] = 0
    row.append(render(y + idle, A=A)); txt.append(None)
    row.append(render(Y + idle, A=A)); txt.append(None)
    row.append(render_heatmap(Y + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_recons_lower(cls, hparams, batch, results, bi, fi, A=256):

    # print_dict("batch", batch)
    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    y_ani = _h("phase_1st.y_ip_rec_low_ani")
    y_aud = _h("phase_1st.y_ip_rec_low_aud")
    Y = _h("Y_IP")
    if y_ani is None or y_aud is None:
        return None

    titles, rows, txts = [], [], []
    titles.append(("Lower Face (Ani)", cls.colors.white))
    titles.append(("Lower Face (aud)", cls.colors.white))
    if Y is not None:
        titles.append(("REAL", cls.colors.green))

    # fmt: off
    # * Row 1
    row, txt = [], []
    row.append(render(y_ani + idle, A=A)); txt.append(None)
    row.append(render(y_aud + idle, A=A)); txt.append(None)
    if Y is not None:
        row.append(render(Y + idle, A=A)); txt.append(None)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_recons_coma(cls, hparams, batch, results, bi, fi, A=256):

    # print_dict("batch", batch)
    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa

    Y = _h("coma.Y")
    y = _h("coma.recons.y")
    y_ctt = _h("coma.recons.y_ctt")
    y_sty = _h("coma.recons.y_sty")

    if Y is not None:
        # copy the non-face part of REAL data
        nidx = results["NON_FACE_VIDX"]
        # y[nidx] = Y[nidx]
        # y_ctt[nidx] = Y[nidx]
        # y_sty[nidx] = Y[nidx]

        titles, rows, txts = [], [], []
        titles.append(("CTT (coma)", cls.colors.white))
        titles.append(("STY (coma)", cls.colors.white))
        titles.append(("Recons (coma)", cls.colors.white))
        titles.append(("REAL (coma)", cls.colors.green))
        titles.append(("Error", cls.colors.red))

        # * Row 1
        row, txt = [], []
        dist_cm = np.linalg.norm(y - Y, axis=-1) * 100
        dist_cm[nidx] = 0
        row.append(render(y_ctt + idle, A=A)); txt.append(None)
        row.append(render(y_sty + idle, A=A)); txt.append(None)
        row.append(render(y + idle, A=A)); txt.append(None)
        row.append(render(Y + idle, A=A)); txt.append(None)
        row.append(render_heatmap(Y + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)
        rows.append(row); txts.append(txt)

        # * draw and return
        canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
        return canvas
    else:
        # return np.zeros((128, 128, 3), dtype=np.uint8)
        return None
    # fmt: on


@register_painter
def demo_recons_audio(cls, hparams, batch, results, bi, fi, A=256):

    # print_dict("batch", batch)
    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    y = _h("phase_1st.y_ip_rec_aud")
    Y = _h("Y_IP")
    if y is None:
        return None
    if Y is None:
        return None

    # copy the non-face part of REAL data
    nidx = results["NON_FACE_VIDX"]
    # y[nidx] = Y[nidx]

    titles, rows, txts = [], [], []
    titles.append(("Recons", cls.colors.white))
    titles.append(("REAL", cls.colors.green))
    titles.append(("Error", cls.colors.red))

    # * Row 1
    # fmt: off
    row, txt = [], []
    dist_cm = np.linalg.norm(y - Y, axis=-1)*100
    dist_cm[nidx] = 0
    row.append(render(y + idle, A=A)); txt.append(None)
    row.append(render(Y + idle, A=A)); txt.append(None)
    row.append(render_heatmap(Y + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_duplex(cls, hparams, batch, results, bi, fi, A=256):

    # print_dict("batch", batch)
    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    Y_IP, Y_JQ = _h("Y_IP"), _h("Y_JQ")
    Y_IQ, Y_JP = _h("Y_IQ"), _h("Y_JP")
    ph1_y_iq, ph1_y_jp = _h("phase_1st.y_iq_swp_ani"), _h("phase_1st.y_jp_swp_ani")
    if ph1_y_iq is None:
        return None
    assert ph1_y_iq is not None and Y_IQ is not None
    assert ph1_y_jp is not None and Y_JP is not None

    ph2_y_ip, ph2_y_jq = _h("phase_2nd.y_ip_cyc_ani"), _h("phase_2nd.y_jq_cyc_ani")
    if ph2_y_ip is not None:
        assert ph2_y_ip is not None and Y_IP is not None
        assert ph2_y_jq is not None and Y_JQ is not None

    # copy the non-face part of REAL data
    nidx = results["NON_FACE_VIDX"]
    # ph1_y_iq[nidx] = Y_IQ[nidx]
    # ph1_y_jp[nidx] = Y_JP[nidx]
    # ph2_y_ip[nidx] = Y_IP[nidx]
    # ph2_y_jq[nidx] = Y_JQ[nidx]

    titles, rows, txts = [], [], []
    titles.append(("Input", cls.colors.green))
    titles.append(("Out  (1st)", cls.colors.white))
    titles.append(("REAL (1st)", cls.colors.green))
    titles.append(("Error(1st)", cls.colors.red))
    if ph2_y_ip is not None:
        titles.append(("Out  (2nd)", cls.colors.white))
        titles.append(("REAL (2nd)", cls.colors.green))
        titles.append(("Error(2nd)", cls.colors.red))

    colors = dict(
        i=(0, 161, 236),
        j=(0, 85, 125),
        p=(0, 236, 137),
        q=(0, 69, 40),
    )

    # fmt: off
    def _txt_tup(ctt, sty, is_real):
        t = f"y_{ctt}{sty}"
        if is_real:
            t = t.upper()
        return (t, (0, 0, 0), f"CTT: {ctt}", colors[ctt], f"STY: {sty}", colors[sty])

    def _append(ctt, sty, pred, REAL):
        dist_cm = np.linalg.norm(pred - REAL, axis=-1) * 100
        dist_cm[nidx] = 0
        row.append(render(pred + idle, A=A)); txt.append(_txt_tup(ctt, sty, False))
        row.append(render(REAL + idle, A=A)); txt.append(None)
        row.append(render_heatmap(REAL + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)

    # * Row 1
    row, txt = [], []
    row.append(render(Y_IP + idle, A=A)); txt.append(_txt_tup("i", "p", True))
    _append("i", "q", ph1_y_iq, Y_IQ)
    if ph2_y_ip is not None:
        _append("i", "p", ph2_y_ip, Y_IP)
    rows.append(row); txts.append(txt)

    # * Row 2
    row, txt = [], []
    row.append(render(Y_JQ + idle, A=A)); txt.append(_txt_tup("j", "q", True))
    _append("j", "p", ph1_y_jp, Y_JP)
    if ph2_y_ip is not None:
        _append("j", "q", ph2_y_jq, Y_JQ)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_duplex_audio(cls, hparams, batch, results, bi, fi, A=256):

    # print_dict("batch", batch)
    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    Y_IP, Y_JQ = _h("Y_IP"), _h("Y_JQ")
    Y_IQ, Y_JP = _h("Y_IQ"), _h("Y_JP")
    ph1_y_iq, ph1_y_jp = _h("phase_1st.y_iq_swp_aud"), _h("phase_1st.y_jp_swp_aud")
    if ph1_y_iq is None:
        return None
    assert ph1_y_iq is not None and Y_IQ is not None
    assert ph1_y_jp is not None and Y_JP is not None

    ph2_y_ip, ph2_y_jq = _h("phase_2nd.y_ip_cyc_aud"), _h("phase_2nd.y_jq_cyc_aud")
    if ph2_y_ip is not None:
        assert ph2_y_ip is not None and Y_IP is not None
        assert ph2_y_jq is not None and Y_JQ is not None

    # copy the non-face part of REAL data
    nidx = results["NON_FACE_VIDX"]
    # ph1_y_iq[nidx] = Y_IQ[nidx]
    # ph1_y_jp[nidx] = Y_JP[nidx]
    # ph2_y_ip[nidx] = Y_IP[nidx]
    # ph2_y_jq[nidx] = Y_JQ[nidx]

    titles, rows, txts = [], [], []
    titles.append(("Input", cls.colors.green))
    titles.append(("Aud+Sty(1st)", cls.colors.white))
    titles.append(("REAL(1st)", cls.colors.green))
    titles.append(("Error(1st)", cls.colors.red))
    if ph2_y_ip is not None:
        titles.append(("Aud+Sty(2nd)", cls.colors.white))
        titles.append(("REAL(2nd)", cls.colors.green))
        titles.append(("Error(2nd)", cls.colors.red))

    colors = dict(
        i=(0, 161, 236),
        j=(0, 85, 125),
        p=(0, 236, 137),
        q=(0, 69, 40),
    )

    # fmt: off
    def _txt_tup(ctt, sty, is_real):
        t = f"y_{ctt}{sty}"
        if is_real:
            t = t.upper()
        return (t, (0, 0, 0), f"AUD: {ctt}", colors[ctt], f"STY: {sty}", colors[sty])

    def _append(ctt, sty, pred, REAL):
        dist_cm = np.linalg.norm(pred - REAL, axis=-1) * 100
        dist_cm[nidx] = 0
        row.append(render(pred + idle, A=A)); txt.append(_txt_tup(ctt, sty, False))
        row.append(render(REAL + idle, A=A)); txt.append(None)
        row.append(render_heatmap(REAL + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)

    # * Row 1
    row, txt = [], []
    row.append(render(Y_IP + idle, A=A)); txt.append(_txt_tup("i", "p", True))
    _append("i", "q", ph1_y_iq, Y_IQ)
    if ph2_y_ip is not None:
        _append("i", "p", ph2_y_ip, Y_IP)
    rows.append(row); txts.append(txt)

    # * Row 2
    row, txt = [], []
    row.append(render(Y_JQ + idle, A=A)); txt.append(_txt_tup("j", "q", True))
    _append("j", "p", ph1_y_jp, Y_JP)
    if ph2_y_ip is not None:
        _append("j", "q", ph2_y_jq, Y_JQ)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_duplex_comp(cls, hparams, batch, results, bi, fi, A=256):

    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    Y_IP, Y_JQ = _h("Y_IP"), _h("Y_JQ")
    Y_IQ, Y_JP = _h("Y_IQ"), _h("Y_JP")
    ph1_y_iq, ph1_y_jp = _h("phase_1st.y_iq_swp_ani"), _h("phase_1st.y_jp_swp_ani")
    if ph1_y_iq is None:
        return None
    assert ph1_y_iq is not None and Y_IQ is not None
    assert ph1_y_jp is not None and Y_JP is not None

    ph2_y_ip, ph2_y_jq = _h("phase_2nd.y_ip_cyc_ani"), _h("phase_2nd.y_jq_cyc_ani")
    if ph2_y_ip is not None:
        assert ph2_y_ip is not None and Y_IP is not None
        assert ph2_y_jq is not None and Y_JQ is not None

    # copy the non-face part of REAL data
    nidx = results["NON_FACE_VIDX"]
    # ph1_y_iq[nidx] = Y_IQ[nidx]
    # ph1_y_jp[nidx] = Y_JP[nidx]
    # ph2_y_ip[nidx] = Y_IP[nidx]
    # ph2_y_jq[nidx] = Y_JQ[nidx]

    ph1_y_i0, ph1_y_0q = _h("phase_1st.y_i0_vis_ani"), _h("phase_1st.y_0q_vis_ani")
    ph2_y_i0, ph2_y_0p = _h("phase_2nd.y_i0_vis_ani"), _h("phase_2nd.y_0p_vis_ani")
    ph1_y_j0, ph1_y_0p = _h("phase_1st.y_j0_vis_ani"), _h("phase_1st.y_0p_vis_ani")
    ph2_y_j0, ph2_y_0q = _h("phase_2nd.y_j0_vis_ani"), _h("phase_2nd.y_0q_vis_ani")

    # fmt: off
    titles, rows, txts = [], [], []
    titles.append(("Input", cls.colors.green))
    titles.append(("CTT  (1st)", cls.colors.white))
    titles.append(("STY  (1st)", cls.colors.white))
    titles.append(("BOTH (1st)", cls.colors.white))
    titles.append(("REAL (1st)", cls.colors.green))
    titles.append(("Error(1st)", cls.colors.red))
    if ph2_y_ip is not None:
        titles.append(("CTT  (2nd)", cls.colors.white))
        titles.append(("STY  (2nd)", cls.colors.white))
        titles.append(("BOTH (2nd)", cls.colors.white))
        titles.append(("REAL (2nd)", cls.colors.green))
        titles.append(("Error(2nd)", cls.colors.red))

    colors = dict(
        i=(0, 161, 236),
        j=(0, 85, 125),
        p=(0, 236, 137),
        q=(0, 69, 40),
    )

    def _txt_tup(tag, is_real=False):
        ctt, sty = tag
        if ctt != "0" and sty == "0":
            return (f"CTT: {ctt}", colors[ctt])
        elif ctt == "0" and sty != "0":
            return (f"STY: {sty}", colors[sty])
        else:
            t = f"y_{ctt}{sty}"
            if is_real:
                t = t.upper()
            return (t, (0, 0, 0), f"CTT: {ctt}", colors[ctt], f"STY: {sty}", colors[sty])

    def _append(tag, pred, REAL):
        dist_cm = np.linalg.norm(pred - REAL, axis=-1) * 100
        dist_cm[nidx] = 0
        row.append(render(pred + idle, A=A)); txt.append(_txt_tup(tag, False))
        row.append(render(REAL + idle, A=A)); txt.append(None)
        row.append(render_heatmap(REAL + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)

    # * Row 1
    row, txt = [], []
    row.append(render(Y_IP + idle, A=A)); txt.append(_txt_tup("ip", True))
    row.append(render(ph1_y_i0 + idle, A=A)); txt.append(_txt_tup("i0"))
    row.append(render(ph1_y_0q + idle, A=A)); txt.append(_txt_tup("0q"))
    _append("iq", ph1_y_iq, Y_IQ)
    if ph2_y_ip is not None:
        row.append(render(ph2_y_i0 + idle, A=A)); txt.append(_txt_tup("i0"))
        row.append(render(ph2_y_0p + idle, A=A)); txt.append(_txt_tup("0p"))
        _append("ip", ph2_y_ip, Y_IP)
    rows.append(row); txts.append(txt)

    # * Row 2
    row, txt = [], []
    row.append(render(Y_JQ + idle, A=A)); txt.append(_txt_tup("jq", True))
    row.append(render(ph1_y_j0 + idle, A=A)); txt.append(_txt_tup("j0"))
    row.append(render(ph1_y_0p + idle, A=A)); txt.append(_txt_tup("0p"))
    _append("jp", ph1_y_jp, Y_JP)
    if ph2_y_ip is not None:
        row.append(render(ph2_y_j0 + idle, A=A)); txt.append(_txt_tup("j0"))
        row.append(render(ph2_y_0q + idle, A=A)); txt.append(_txt_tup("0q"))
        _append("jq", ph2_y_jq, Y_JQ)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_duplex_comp_audio(cls, hparams, batch, results, bi, fi, A=256):

    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    Y_IP, Y_JQ = _h("Y_IP"), _h("Y_JQ")
    Y_IQ, Y_JP = _h("Y_IQ"), _h("Y_JP")
    ph1_y_iq, ph1_y_jp = _h("phase_1st.y_iq_swp_aud"), _h("phase_1st.y_jp_swp_aud")
    if ph1_y_iq is None:
        return None
    assert ph1_y_iq is not None and Y_IQ is not None
    assert ph1_y_jp is not None and Y_JP is not None

    ph2_y_ip, ph2_y_jq = _h("phase_2nd.y_ip_cyc_aud"), _h("phase_2nd.y_jq_cyc_aud")
    if ph2_y_ip is not None:
        assert ph2_y_ip is not None and Y_IP is not None
        assert ph2_y_jq is not None and Y_JQ is not None

    # copy the non-face part of REAL data
    nidx = results["NON_FACE_VIDX"]
    # ph1_y_iq[nidx] = Y_IQ[nidx]
    # ph1_y_jp[nidx] = Y_JP[nidx]
    # ph2_y_ip[nidx] = Y_IP[nidx]
    # ph2_y_jq[nidx] = Y_JQ[nidx]

    ph1_y_i0, ph1_y_0q = _h("phase_1st.y_i0_vis_aud"), _h("phase_1st.y_0q_vis_ani")
    ph2_y_i0, ph2_y_0p = _h("phase_2nd.y_i0_vis_aud"), _h("phase_2nd.y_0p_vis_ani")
    ph1_y_j0, ph1_y_0p = _h("phase_1st.y_j0_vis_aud"), _h("phase_1st.y_0p_vis_ani")
    ph2_y_j0, ph2_y_0q = _h("phase_2nd.y_j0_vis_aud"), _h("phase_2nd.y_0q_vis_ani")

    # fmt: off
    titles, rows, txts = [], [], []
    titles.append(("Input", cls.colors.green))
    titles.append(("AUD  (1st)", cls.colors.white))
    titles.append(("STY  (1st)", cls.colors.white))
    titles.append(("BOTH (1st)", cls.colors.white))
    titles.append(("REAL (1st)", cls.colors.green))
    titles.append(("Error(1st)", cls.colors.red))
    if ph2_y_ip is not None:
        titles.append(("AUD  (2nd)", cls.colors.white))
        titles.append(("STY  (2nd)", cls.colors.white))
        titles.append(("BOTH (2nd)", cls.colors.white))
        titles.append(("REAL (2nd)", cls.colors.green))
        titles.append(("Error(2nd)", cls.colors.red))

    colors = dict(
        i=(0, 161, 236),
        j=(0, 85, 125),
        p=(0, 236, 137),
        q=(0, 69, 40),
    )

    def _txt_tup(tag, is_real=False):
        ctt, sty = tag
        if ctt != "0" and sty == "0":
            return (f"AUD: {ctt}", colors[ctt])
        elif ctt == "0" and sty != "0":
            return (f"STY: {sty}", colors[sty])
        else:
            t = f"y_{ctt}{sty}"
            if is_real:
                t = t.upper()
            return (t, (0, 0, 0), f"AUD: {ctt}", colors[ctt], f"STY: {sty}", colors[sty])

    def _append(tag, pred, REAL):
        dist_cm = np.linalg.norm(pred - REAL, axis=-1) * 100
        dist_cm[nidx] = 0
        row.append(render(pred + idle, A=A)); txt.append(_txt_tup(tag, False))
        row.append(render(REAL + idle, A=A)); txt.append(None)
        row.append(render_heatmap(REAL + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)

    # * Row 1
    row, txt = [], []
    row.append(render(Y_IP + idle, A=A)); txt.append(_txt_tup("ip", True))
    row.append(render(ph1_y_i0 + idle, A=A)); txt.append(_txt_tup("i0"))
    row.append(render(ph1_y_0q + idle, A=A)); txt.append(_txt_tup("0q"))
    _append("iq", ph1_y_iq, Y_IQ)
    if ph2_y_ip is not None:
        row.append(render(ph2_y_i0 + idle, A=A)); txt.append(_txt_tup("i0"))
        row.append(render(ph2_y_0p + idle, A=A)); txt.append(_txt_tup("0p"))
        _append("ip", ph2_y_ip, Y_IP)
    rows.append(row); txts.append(txt)

    # * Row 2
    row, txt = [], []
    row.append(render(Y_JQ + idle, A=A)); txt.append(_txt_tup("jq", True))
    row.append(render(ph1_y_j0 + idle, A=A)); txt.append(_txt_tup("j0"))
    row.append(render(ph1_y_0p + idle, A=A)); txt.append(_txt_tup("0p"))
    _append("jp", ph1_y_jp, Y_JP)
    if ph2_y_ip is not None:
        row.append(render(ph2_y_j0 + idle, A=A)); txt.append(_txt_tup("j0"))
        row.append(render(ph2_y_0q + idle, A=A)); txt.append(_txt_tup("0q"))
        _append("jq", ph2_y_jq, Y_JQ)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_swap_cycle(cls, hparams, batch, results, bi, fi, A=256):

    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    Y_IP = _h("Y_IP")
    ph1_y_iq = _h("phase_1st.y_iq_swp")
    ph2_y_ip = _h("phase_2nd.y_ip_cyc")
    if ph1_y_iq is None or ph2_y_ip is None:
        return None
    assert Y_IP is not None
    assert ph1_y_iq is not None
    assert ph2_y_ip is not None

    # fmt: off
    titles, rows, txts = [], [], []
    titles.append(("Input", cls.colors.green))
    titles.append(("Out (swap once)", cls.colors.white))
    titles.append(("Out (swap twice)", cls.colors.white))

    # * Row 1
    row, txt = [], []
    row.append(render(Y_IP + idle, A=A)); txt.append(None)
    row.append(render(ph1_y_iq + idle, A=A)); txt.append(None)
    row.append(render(ph2_y_ip + idle, A=A)); txt.append(None)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas


@register_painter
def demo_audio_style(cls, hparams, batch, results, bi, fi, A=256):

    # print_dict("batch", batch)
    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    y = _h("pred")
    y_aud = _h("only_aud")
    y_sty = _h("only_sty")
    Y = _h("Y")
    if y is None:
        return None

    # copy the non-face part of REAL data
    nidx = results["NON_FACE_VIDX"]
    # y[nidx] = Y[nidx]

    titles, rows, txts = [], [], []
    titles.append(("Pred", cls.colors.white))
    if Y is not None:
        titles.append(("REAL", cls.colors.green))
        titles.append(("Error", cls.colors.red))
    if y_aud is not None:
        titles.append(("Only Audio", cls.colors.white))
    if y_sty is not None:
        titles.append(("Only Style", cls.colors.white))

    # * Row 1
    # fmt: off
    row, txt = [], []
    row.append(render(y + idle, A=A)); txt.append(None)
    if Y is not None:
        dist_cm = np.linalg.norm(y - Y, axis=-1)*100
        dist_cm[nidx] = 0
        row.append(render(Y + idle, A=A)); txt.append(None)
        row.append(render_heatmap(Y + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)
    if y_aud is not None:
        row.append(render(y_aud + idle, A=A)); txt.append(None)
    if y_sty is not None:
        row.append(render(y_sty + idle, A=A)); txt.append(None)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas
