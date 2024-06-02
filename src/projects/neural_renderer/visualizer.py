import numpy as np

from src import metrics
from src.engine.painter import draw_canvas, heatmap
from src.projects.anim import register_painter


@register_painter
def draw_nr_inter(cls, _, batch, results, bi, fi, A=256):

    # guard if no fake or target image exists
    if results.get("nr_fake_inter") is None or batch.get("image") is None:
        return np.zeros((A, A, 3), dtype=np.uint8)  # type: ignore

    # check shape
    if results["nr_fake_inter"].ndim == 4:
        assert batch["image"].shape[1] == 1 and fi == 0

    def _h(key):
        if key not in results:
            return None
        if results["nr_fake_inter"].ndim == 4:
            tensor = results[key]
            if len(tensor) == 1:
                ret = tensor[0].detach().cpu().numpy()
            else:
                ret = tensor[bi].detach().cpu().numpy()
        else:
            tensor = results[key]
            if tensor.shape[0] == 1:
                ret = results[key][0, fi].detach().cpu().numpy()
            else:
                ret = results[key][bi, fi].detach().cpu().numpy()
        return ret.transpose(1, 2, 0)

    def _b(key):
        return batch[key][bi, fi].cpu().numpy().transpose(1, 2, 0)

    # get from results
    fake = _h("nr_fake_inter")
    fake = cls.may_CHW2HWC(fake)

    # get real
    real_full = _b("image")
    mask_mouth = _b("mouth_mask")
    mask = _h("mask")
    real = cls.masked(real_full, mask)

    # ajdust range
    fake = (fake + 1.0) / 2.0
    real = (real + 1.0) / 2.0

    if mask_mouth is not None:
        mask_mouth = np.repeat(mask_mouth.astype(np.float32), 3, axis=-1)

    # optional
    tex3 = _h("nr_tex3")
    if tex3 is not None:
        tex3 = (tex3 + 1.0) / 2.0
    nr_tex = _h("nr_tex_face")
    if nr_tex is not None:
        nr_tex = (nr_tex[..., :3] + 1.0) / 2.0
    nr_tex_mouth = _h("nr_tex_mouth")
    if nr_tex_mouth is not None:
        nr_tex_mouth = (nr_tex_mouth[..., :3] + 1.0) / 2.0

    # canvas
    imgs, txts = [], []
    # row 0
    row, txt = [], []
    row.append(real)
    txt.append(("Ground Truth (maksed)", None))
    row.append(fake)
    txt.append(("Neural Renderer (intermediate)", None))
    if mask_mouth is not None:
        row.append(mask_mouth)
        txt.append(("Mask Mouth", None))
    if tex3 is not None:
        row.append(tex3)
        txt.append(("Sampled Texture (3ch)", None))
    if nr_tex is not None:
        row.append(nr_tex)
        txt.append(("Neural Texture (3ch)", None))
    if nr_tex_mouth is not None:
        row.append(nr_tex_mouth)
        txt.append(("Neural Texture Mouth (3ch)", None))
    # append row 0
    imgs.append(row)
    txts.append(txt)

    canvas = draw_canvas(imgs, txts, A)
    return canvas


@register_painter
def draw_nr(cls, _, batch, results, bi, fi, A=256):

    # guard if no fake or target image exists
    if results.get("nr_fake") is None or batch.get("image") is None:
        return np.zeros((A, A, 3), dtype=np.uint8)  # type: ignore

    # check shape
    if results["nr_fake"].ndim == 4:
        assert batch["image"].shape[1] == 1 and fi == 0

    psnr = metrics.psnr(results["nr_fake"], batch["image"])
    str_psnr = f"PSNR: {float(psnr.mean()):.1f}"  # type: ignore

    def _h(key):
        if key not in results:
            return None
        if results["nr_fake"].ndim == 4:
            tensor = results[key]
            if len(tensor) == 1:
                return tensor[0].detach().cpu().numpy()
            else:
                return tensor[bi].detach().cpu().numpy()
        else:
            tensor = results[key]
            if tensor.shape[0] == 1:
                return results[key][0, fi].detach().cpu().numpy()
            else:
                return results[key][bi, fi].detach().cpu().numpy()

    def _b(key):
        return batch[key][bi, fi].cpu().numpy()

    # get from results
    fake = _h("nr_fake")
    fake = cls.may_CHW2HWC(fake)

    # get real
    real = _b("image")
    real = cls.may_CHW2HWC(real)

    # ajdust range
    fake = (fake + 1.0) / 2.0
    real = (real + 1.0) / 2.0
    delta = heatmap(np.linalg.norm(fake - real, axis=-1), vmin=0, vmax=1, colorbar=True)  # type: ignore

    # canvas
    imgs, txts = [], []
    # row 0
    row, txt = [], []
    row.append(real)
    txt.append(("Ground Truth", None))
    row.append(fake)
    txt.append(("Neural Renderer", None, str_psnr, None))
    row.append(delta)
    txt.append(("Pixel Error", None))
    # append row 0
    imgs.append(row)
    txts.append(txt)

    canvas = draw_canvas(imgs, txts, A)
    return canvas
