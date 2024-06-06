import os
from shutil import copytree, copyfile

animnet_subdirs = {
    "animnet-cmb3d": "ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4",
    "animnet-decmp-abl_e2e": "ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4-BatchSamplerWithDtwPair",
    "animnet-decmp-abl_no_reg": "ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4-BatchSamplerWithDtwPair",
    "animnet-decmp-abl_no_reg_no_cyc": "ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4-BatchSamplerWithDtwPair",
    "animnet-decmp-abl_no_reg_no_dtw": "ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4-BatchSamplerWithDtwPair",
    "animnet-decmp-abl_no_reg_no_swp": "ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4-BatchSamplerWithDtwPair",
    "animnet-refer": "ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4",
    "animnet-track": "ds16_xfmr-conv_causal-blend50_trainable-seq20-bsz4",
}

nr_subdirs = {
    "f001_clinton": "pix2pix-tex256_16-in-bs8-lr0.002-aug_flip-loss_fake_3.0_1.0_face_1.0_1.0_tex3_1.0_1.0",
    "m000_obama": "pix2pix-tex256_16-in-bs8-lr0.002-aug_flip-loss_fake_3.0_1.0_face_1.0_1.0_tex3_1.0_1.0",
    "m001_trump": "pix2pix-tex256_16-in-bs8-lr0.002-noaug-loss_fake_3.0_1.0_face_1.0_1.0_tex3_1.0_1.0",
    "m002_taruvinga": "pix2pix-pix2pix_3-tex256_16-in-bs8-lr0.002-aug_flip-loss_fake_3.0_1.0_face_1.0_1.0_tex3_1.0_1.0",
    "m003_iphoneXm0": "pix2pix-pix2pix_3-tex256_16-in-bs8-lr0.002-aug_flip-loss_fake_3.0_1.0_face_1.0_1.0_tex3_1.0_1.0",
}



def process_anime(src_root: str, tar_root: str):

    for src_exp_name in [
        "animnet-cmb3d",
        "animnet-decmp-abl_e2e",
        "animnet-decmp-abl_no_reg",
        "animnet-decmp-abl_no_reg_no_cyc",
        "animnet-decmp-abl_no_reg_no_dtw",
        "animnet-decmp-abl_no_reg_no_swp",
        "animnet-refer",
        "animnet-track",
    ]:
        if src_exp_name == "animnet-decmp-abl_no_reg":
            tar_exp_name = "animnet-decmp"
        elif src_exp_name.startswith("animnet-decmp-abl_no_reg_"):
            tar_exp_name = src_exp_name.replace("animnet-decmp-abl_no_reg_", "animnet-decmp-abl_")
        else:
            tar_exp_name = src_exp_name
        
        tar_dir = os.path.join(tar_root, tar_exp_name)
        if os.path.exists(os.path.join(tar_dir, "tb/hparams.yaml")):
            print(f"Skip copied {src_exp_name}")
            continue

        src_dir = os.path.join(src_root, src_exp_name, animnet_subdirs[src_exp_name])
        assert os.path.isabs(src_dir), f"Failed to find dir: {src_dir}"

        # Copy.
        os.makedirs(os.path.join(tar_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(tar_dir, "tb"), exist_ok=True)
        for fname in ("checkpoints/epoch_50.pth", "tb/hparams.yaml"):
            copyfile(
                os.path.join(src_dir, fname),
                os.path.join(tar_dir, fname),
            )
        print(f"Done {src_exp_name}")


def process_nr(src_dir: str, tar_dir: str):
    if os.path.exists(os.path.join(tar_dir, "tb/hparams.yaml")):
        print(f"Skip copied {tar_dir}")
        return

    assert os.path.isabs(src_dir), f"Failed to find dir: {src_dir}"

    # Copy.
    os.makedirs(os.path.join(tar_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(tar_dir, "tb"), exist_ok=True)
    for fname in ("checkpoints/epoch_60.pth", "tb/hparams.yaml"):
        copyfile(
            os.path.join(src_dir, fname),
            os.path.join(tar_dir, fname),
        )
    print(f"Done {tar_dir}")


# Copy pretrain models.
for spk in ["m001_trump", "f001_clinton", "m000_obama", "m002_taruvinga", "m003_iphoneXm0"]:
    src_root = f"/media/chaiyujin/Yuki-SSD2/Project2021/stylized-sa/runs/fps25-face_noeyeballs/{spk}/"
    tar_root = f"/home/chaiyujin/Documents/GitHub/psfa-2022/runs/anime/{spk}/"
    process_anime(src_root, tar_root)

    src_root = f"/media/chaiyujin/Yuki-SSD2/Project2021/stylized-sa/runs/neural_renderer/{spk}/{nr_subdirs[spk]}"
    tar_root = f"/home/chaiyujin/Documents/GitHub/psfa-2022/runs/neural_renderer/{spk}/"
    process_nr(src_root, tar_root)


# Copy datasets.
for spk in ["m001_trump", "f001_clinton", "m000_obama", "m002_taruvinga", "m003_iphoneXm0"]:
    src_root = f"/media/chaiyujin/Yuki-SSD2/Project2021/stylized-sa/data/datasets/talk_video/celebtalk/data/{spk}"
    tar_root = f"/home/chaiyujin/Documents/GitHub/psfa-2022/runs/neural_renderer/{spk}/"
