from typing import Optional

from src.engine.seq_ops import fold, unfold_dict
from src.modules.neural_renderer import NeuralRenderer


def fw_animnet(animnet, batch, **kwargs):
    """The animnet model will generate the stylized offsets. The output will be organized in dict."""

    # * inputs
    idle_verts = batch.get("idle_verts")
    audio_dict = batch.get("audio_dict", dict())
    assert idle_verts is not None

    # * animnet model
    stylized_out = animnet(
        idle_verts=idle_verts,
        audio_dict=audio_dict,
        style_id=batch.get("style_id"),
        **kwargs,
    )
    code_dict = stylized_out["code_dict"]

    # prepare the output verts_dict
    idle_verts = idle_verts.unsqueeze(1)
    trck_delta = batch.get("offsets_tracked")
    REAL_delta = batch.get("offsets_REAL")

    ret = dict(
        idle_verts=idle_verts,
        trck_verts=(idle_verts + trck_delta) if trck_delta is not None else None,
        REAL_verts=(idle_verts + REAL_delta) if REAL_delta is not None else None,
        dfrm_verts=stylized_out["out_verts"],
        dfrm_delta=stylized_out["out_verts"] - idle_verts,
        code_dict=code_dict,
    )

    # check shape
    for k in ["trck_verts", "REAL_verts"]:
        assert ret[k] is None or ret[k].shape == ret["dfrm_verts"].shape

    return ret


def fw_renderer(renderer: Optional[NeuralRenderer], verts, batch, optim_visibility=False):
    if renderer is None:
        return dict()

    rotat = batch["code_dict"]["rotat"]
    transl = batch["code_dict"]["transl"]
    cam = batch["code_dict"]["cam"]

    assert rotat.shape[1] == verts.shape[1]
    assert transl.shape[1] == verts.shape[1]
    assert cam.shape[1] == verts.shape[1]
    # print(verts.shape, rotat.shape)

    verts, frm = fold(verts)
    rotat, _ = fold(rotat)
    transl, _ = fold(transl)
    cam, _ = fold(cam)

    ret = renderer(
        clip_sources=batch["clip_source"],
        vertices=verts,
        rotat=rotat,
        transl=transl,
        cam=cam,
    )

    v_rigid = renderer.rigid(verts, rotat=rotat, transl=transl)
    lmks_fw75 = renderer.extension.get_static_fw75_landmarks(v_rigid)
    cntr_pts, cntr_mask = renderer.extension.find_contour(v_rigid, ret["rast_out"])
    # * project landmarks
    lmks_fw75 = renderer.camera.transform(lmks_fw75)
    cntr_pts = renderer.camera.transform(cntr_pts)
    # * correct contour landmarks for fw75
    if batch.get("lmks_fw75") is not None:
        real_lmks_fw75 = batch["lmks_fw75"]
        if frm is not None:
            real_lmks_fw75, _ = fold(real_lmks_fw75)
        lmks_fw75 = renderer.extension.correct_fw75_contour(lmks_fw75, cntr_pts, cntr_mask, real_lmks_fw75)
    ret["lmks_fw75"] = lmks_fw75
    ret["cntr_pts"] = cntr_pts

    return unfold_dict(ret, frm)
