from dataclasses import dataclass
from typing import Optional


@dataclass
class PathConf:
    # path to original working directory
    # hydra hijacks working directory by changing it to the current log directory,
    # so it's useful to have this path as a special variable
    # learn more here: https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
    root_dir: str = "${hydra:runtime.cwd}"
    generate_dir: Optional[str] = None

    # path to folder with data
    assets_dir: str = "${path.root_dir}/assets"
    dataset_dir: str = "${path.root_dir}/assets/datasets"
    data_dir: str = "${path.root_dir}/src/data"
    archived_dir: str = "${path.root_dir}/archived"
    snaps_dir: str = "${path.root_dir}/.snaps"
    runs_dir: str = "${path.root_dir}/runs"

    # the path to store experiment logs
    exp_dir: str = "${path.root_dir}/.snaps/run"  # ${now:%Y-%m-%d}/${now:%H-%M-%S}
