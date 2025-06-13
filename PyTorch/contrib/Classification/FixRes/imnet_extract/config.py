# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted to tecorigin hardware
from typing import NamedTuple


class ClusterConfig(NamedTuple):
    dist_backend: str
    dist_url: str


class TrainerConfig(NamedTuple):
    data_folder: str
    architecture: str
    weight_path: str
    dataset_path: str
    save_path: str
    workers: int
    input_size: int
    batch_per_gpu: int
    local_rank: int
    global_rank: int
    num_tasks: int
    job_id: str
    save_folder: str
