import argparse

import numpy as np

from src.data.mesh import load_mesh

from .api import render_ours
from .utils import interpolate_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--iden_path", type=str, required=True)
    parser.add_argument("--offsets_npy", type=str, required=True)
    parser.add_argument("--reenact_video", type=str, required=True)
    parser.add_argument("--reenact_coeff", type=str, required=True)
    parser.add_argument("--nr_ckpt", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--reenact_static_frame", type=int)
    args = parser.parse_args()

    idle_verts = load_mesh(args.iden_path)[0]
    verts = np.load(args.offsets_npy) + idle_verts[None]

    render_ours(
        args.out_path,
        verts,
        idle_verts,
        args.reenact_video,
        args.reenact_coeff,
        args.nr_ckpt,
        audio_fpath=args.audio_path,
        static_frame=args.reenact_static_frame,
        need_metrics=True,
    )
