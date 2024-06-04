if __name__ == "__main__":
    import os
    from glob import glob

    import numpy as np

    from src.data.mesh import load_mesh

    from .api import render_ours
    from .utils import interpolate_features

    # Identity
    idle_verts = load_mesh("assets/datasets/talk_video/celebtalk/data/m001_trump/fitted/identity/identity.obj")[0]
    # Mesh frames.
    npy_files = glob("assets/datasets/talk_video/celebtalk/data/m001_trump/fitted/vld-000/meshes/*.npy")
    npy_files = sorted(npy_files)
    verts = np.asarray([np.load(x) for x in npy_files], dtype=np.float32)  # type: ignore
    verts = interpolate_features(verts, 30, 25)

    # npy_file = "/run/user/1000/gvfs/sftp:host=10.76.2.227/home/chaiyujin/Documents/Project2021/stylized-sa/runs/fps25-face_noeyeballs/m001_trump/animnet-decmp-abl_no_reg/ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4-BatchSamplerWithDtwPair/generated/[50][test]m001_trump/vld-000/dump-offsets-final.npy"
    # verts = np.load(npy_file) + idle_verts[None]

    reenact_video = "assets/datasets/talk_video/celebtalk/data/m001_trump/vld-000/images"
    reenact_coeff = "assets/datasets/talk_video/celebtalk/data/m001_trump/fitted/vld-000"
    model_path = "runs/neural_renderer/m001_trump/pix2pix-tex256_16-in-bs8-lr0.002-noaug-loss_fake_3.0_1.0_face_1.0_1.0_tex3_1.0_1.0/checkpoints/epoch_60.pth"
    audio_fpath = "data/datasets/talk_video/celebtalk/data/m001_trump/vld-000/audio.wav"

    render_ours(
        "./test_out.mp4",
        verts,
        idle_verts,
        reenact_video,
        reenact_coeff,
        model_path,
        audio_fpath=audio_fpath,
        need_metrics=True,
    )
