import numpy as np

from src.data import get_vocaset_template_vertices
from src.engine.mesh_renderer import render, render_heatmap
from src.engine.painter import Text, draw_canvas, put_texts
from src.projects.anim.vis import register_painter


@register_painter
def demo_columns(cls, hparams, batch, results, bi, fi, A=256):

    # print_dict("batch", batch)
    idle = batch["idle_verts"][bi].detach().cpu().numpy()

    # fmt: off
    def _b(key, d=bi): return cls.get(batch,   key, d, fi)  # noqa
    def _h(key, d=bi): return cls.get(results, key, d, fi)  # noqa
    # fmt: on

    y = _h("animnet.dfrm_delta")
    r = _h("animnet.refr_delta")
    Y = _h("animnet.REAL_delta")
    if Y is None:
        Y = _h("animnet.trck_delta")

    # copy the non-face part of REAL data
    nidx = results["NON_FACE_VIDX"]
    # y[nidx] = Y[nidx]

    titles, rows, txts = [], [], []
    titles.append(("Recons", cls.colors.white))
    if Y is not None:
        titles.append(("REAL", cls.colors.green))
        titles.append(("Error", cls.colors.red))
    if r is not None:
        titles.append(("Refer", cls.colors.white))

    # fmt: off
    # * Row 1
    row, txt = [], []
    row.append(render(y + idle, A=A)); txt.append(None)
    if Y is not None:
        dist_cm = np.linalg.norm(y - Y, axis=-1) * 100
        dist_cm[nidx] = 0
        row.append(render(Y + idle, A=A)); txt.append(None)
        row.append(render_heatmap(Y + idle, dist_cm, A, 0, 1, 'cm')); txt.append(None)
    if r is not None:
        row.append(render(r + idle, A=A)); txt.append(None)
    rows.append(row); txts.append(txt)
    # fmt: on

    # * draw and return
    canvas = draw_canvas(rows, txts, A=A, titles=titles, shrink_columns="all", shrink_ratio=0.75)
    return canvas
