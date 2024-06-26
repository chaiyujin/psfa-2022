# PSFA
PyTorch Implementation of our paper ["Personalized Audio-Driven 3D Facial Animation via Style-Content Disentanglement"](https://ieeexplore.ieee.org/document/9992151/) published in IEEE TVCG.
Please cite our paper if you use or adapt from this repo.

You can also access the [Project Page](https://chaiyujin.github.io/psfa/) for supplementary videos.

## Dependencies
- Software & Packages
  - Python 3.7~3.9
  - boost: `apt install boost` or `brew install boost`
  - [chaiyujin/videoio-python](https://github.com/chaiyujin/videoio-python)
  - [NVlabs/nvdiffrast](https://github.com/NVlabs/nvdiffrast.git)
  - pytorch >= 1.7.1 (Also tested with 2.0.1).
  - tensorflow >= 1.15.3 (Also tested with 2.13.0).
  - [torch_geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
  - Install other dependencies with `pip install -r requirements.txt`. Pytorch-lightning changes API frequently, thus pytorch-lightning==1.5.8 must be used.
- 3rd-party Models
  - Download [deepspeech-0.1.0-models](https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz) and unwrap it into `./assets/pretrain_models/deepspeech-0.1.-models/`.
  - FLAME: Download from [official website](https://flame.is.tue.mpg.de/) and put model at `assets/flame-data/FLAME2020/generic_model.pkl` and masks at `assets/flame-data/FLAME_masks/FLAME_masks.pkl`.
    - After downloading, convert chumpy model to numpy version by: `python assets/flame-data/FLAME2020/to_numpy.py`. Then, you can get `generic_model-np.pkl` in the same folder.

## Generate animation with pre-trained models
1. Download pre-trained models and data from [Google Drive](https://drive.google.com/drive/folders/1Xoof9j5-q8c42gs87IxBTMcxIyUkTfn9?usp=sharing) and put them at the correct directories. The dataset files are compressed as `.7z` files, which should be uncompressed.

1. Modify and run `bash scripts/generate.sh` to generate new animations.

## Training
All data-processing and training codes are contained, but not cleaned yet.

## Citation
```
@article{chai2024personalized,
  author={Chai, Yujin and Shao, Tianjia and Weng, Yanlin and Zhou, Kun},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  title={Personalized audio-driven 3d facial animation via style-content disentanglement},
  year={2024},
  volume={30},
  number={3},
  pages={1803-1820},
  doi={10.1109/TVCG.2022.3230541}
}
```
