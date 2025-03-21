#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import datasets
import pandas as pd
from datasets import Features, Sequence, Value
from torch.utils.data import Dataset

from uni2ts.common.env import env
from uni2ts.common.typing import GenFunc

# from ._base import DatasetBuilder
from uni2ts.data.builder._base import DatasetBuilder
from uni2ts.data.dataset import (
    EvalDataset,
    FinetuneDataset,
    SampleTimeSeriesType,
    TimeSeriesDataset,
)
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Transformation

from uni2ts.data.builder.simple import SimpleFinetuneDatasetBuilder, SimpleEvalDatasetBuilder, OnlineValDatasetBuilder


def generate_online_builder(
    dataset: str,
    offset: int,
    eval_length: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    mode: str,
    storage_path: Path = env.CUSTOM_DATA_PATH,
) -> SimpleEvalDatasetBuilder:

    distance = 1
    windows = (eval_length - prediction_length) // distance + 1

    # distance = prediction_length
    # windows = eval_length // prediction_length

    return SimpleEvalDatasetBuilder(
        dataset=dataset,
        offset=offset,
        windows=windows,
        distance=distance,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        mode=mode,
        storage_path=storage_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["wide", "wide_multivariate"],
        default="wide",
    )
    # Define the `freq` argument with a default value. Use this value as 'freq' if 'freq' is None.
    parser.add_argument(
        "--freq",
        default="H",  # Set the default value
        help="The user specified frequency",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Only the part before offset will be used for experiments"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize train and eval data with train statistics",
    )

    args = parser.parse_args()

    # Same as SimpleFinetuneDatasetBuilder. Just use ratio as offset. Use 0.2 for warmup.
    # 存到lsf文件夹了..
    df = pd.read_csv(Path(args.file_path), index_col=0, parse_dates=True)
    total_length = len(df) if args.offset is None else args.offset
    warmup_offset = int(total_length * 0.2)
    val_offset = int(total_length * 0.25)

    # Create Warmup dataset. Only contains the part before warmup_offset.
    warmup_dataset_builder = SimpleFinetuneDatasetBuilder(
        dataset=f"{args.dataset_name}_online_warmup",
        windows=None,
        distance=None,
        prediction_length=None,
        context_length=None,
        patch_size=None,
    )
    warmup_dataset_builder.build_dataset(
        file=Path(args.file_path),
        dataset_type=args.dataset_type,
        offset=warmup_offset,
        freq=args.freq,
        normalize=True     # args.normalize,
    )

    # Create val dataset. Contains the part before val_offset, i.e. train+val.
    OnlineValDatasetBuilder(
        dataset=f"{args.dataset_name}_online_val" if args.normalize else f"{args.dataset_name}_online_val_wo_norm",
        offset=None,
        windows=None,
        distance=None,
        prediction_length=None,
        context_length=None,
        patch_size=None,
    ).build_dataset(
        file=Path(args.file_path),
        dataset_type=args.dataset_type,
        offset=val_offset,
        freq=args.freq,
        mean=None,  # warmup_dataset_builder.mean,
        std=None,   # warmup_dataset_builder.std,
    )

    # Online dataset: Use full dataset. Set 'offset' as val_offset in cli config to indicate the start of online stage.
    SimpleEvalDatasetBuilder(
        dataset=f"{args.dataset_name}_online" if args.normalize else f"{args.dataset_name}_online_wo_norm",
        offset=None,
        windows=None,
        distance=None,
        prediction_length=None,
        context_length=None,
        patch_size=None,
    ).build_dataset(
        file=Path(args.file_path),
        dataset_type=args.dataset_type,
        freq=args.freq,
        mean=None,  # warmup_dataset_builder.mean,
        std=None   # warmup_dataset_builder.std,
    )

    if not args.normalize:
        import pickle
        stats_path = warmup_dataset_builder.storage_path / 'online_warmup_statistics' / f"{args.dataset_name}.pkl"
        stats_path.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
        with open(stats_path, "wb") as f:
            pickle.dump({"mean": warmup_dataset_builder.mean, "std": warmup_dataset_builder.std}, f)

