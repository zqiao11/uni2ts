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
from uni2ts.data.dataset import (
    EvalDataset,
    FinetuneDataset,
    SampleTimeSeriesType,
    TimeSeriesDataset,
)
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Transformation

# from ._base import DatasetBuilder
from uni2ts.data.builder._base import DatasetBuilder

def _from_long_dataframe(
    df: pd.DataFrame,
    offset: Optional[int] = None,
    date_offset: Optional[pd.Timestamp] = None,
    freq: str = "H",
) -> tuple[GenFunc, Features]:
    items = df.item_id.unique()

    # Infer the freq and generate the prompt
    inferred_freq = pd.infer_freq(df.index)

    if inferred_freq is not None:
        print(
            f"Inferred frequency: {inferred_freq}. Using this value for the 'freq' parameter."
        )
    else:
        print(
            f"Inferred frequency is None. Using predefined {freq} for the 'freq' parameter."
        )

    def example_gen_func() -> Generator[dict[str, Any], None, None]:
        for item_id in items:
            item_df = df.query(f'item_id == "{item_id}"').drop("item_id", axis=1)
            if offset is not None:
                item_df = item_df.iloc[:offset]
            elif date_offset is not None:
                item_df = item_df[item_df.index <= date_offset]
            yield {
                "target": item_df.to_numpy(),
                "start": item_df.index[0],
                "freq": (
                    pd.infer_freq(df.index)
                    if pd.infer_freq(df.index) is not None
                    else freq
                ),
                "item_id": item_id,
            }

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Value("float32")),
        )
    )

    return example_gen_func, features


def _from_wide_dataframe(
    df: pd.DataFrame,
    offset: Optional[int] = None,
    date_offset: Optional[pd.Timestamp] = None,
    freq: str = "H",
) -> tuple[GenFunc, Features]:
    if offset is not None:
        df = df.iloc[:offset]
    elif date_offset is not None:
        df = df[df.index <= date_offset]

    print(df)

    # Infer the freq and generate the prompt
    inferred_freq = pd.infer_freq(df.index)

    if inferred_freq is not None:
        print(
            f"Inferred frequency: {inferred_freq}. Using this value for the 'freq' parameter."
        )
    else:
        print(
            f"Inferred frequency is None. Using predefined {freq} for the 'freq' parameter."
        )

    def example_gen_func() -> Generator[dict[str, Any], None, None]:
        for i in range(len(df.columns)):
            yield {
                "target": df.iloc[:, i].to_numpy(),
                "start": df.index[0],
                "freq": (
                    pd.infer_freq(df.index)
                    if pd.infer_freq(df.index) is not None
                    else freq
                ),
                "item_id": f"item_{i}",
            }

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Value("float32")),
        )
    )

    return example_gen_func, features


def _from_wide_dataframe_multivariate(
    df: pd.DataFrame,
    offset: Optional[int] = None,
    date_offset: Optional[pd.Timestamp] = None,
    freq: str = "H",
) -> tuple[GenFunc, Features]:
    if offset is not None:
        df = df.iloc[:offset]
    elif date_offset is not None:
        df = df[df.index <= date_offset]

    # Infer the freq and generate the prompt
    inferred_freq = pd.infer_freq(df.index)

    if inferred_freq is not None:
        print(
            f"Inferred frequency: {inferred_freq}. Using this value for the 'freq' parameter."
        )
    else:
        print(
            f"Inferred frequency is None. Using predefined {freq} for the 'freq' parameter."
        )

    def example_gen_func() -> Generator[dict[str, Any], None, None]:
        yield {
            "target": df.to_numpy().T,
            "start": df.index[0],
            "freq": (
                pd.infer_freq(df.index) if pd.infer_freq(df.index) is not None else freq
            ),
            "item_id": "item_0",
        }

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Sequence(Value("float32")), length=len(df.columns)),
        )
    )

    return example_gen_func, features


@dataclass
class SimpleDatasetBuilder(DatasetBuilder):
    dataset: str
    weight: float = 1.0
    sample_time_series: Optional[SampleTimeSeriesType] = SampleTimeSeriesType.NONE
    storage_path: Path = env.CUSTOM_DATA_PATH
    mean = None
    std = None

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        offset: Optional[int] = None,
        date_offset: Optional[pd.Timestamp] = None,
        freq: str = "H",
        normalize: Optional[bool] = False,
    ):
        assert offset is None or date_offset is None, (
            "One or neither offset and date_offset must be specified, but not both. "
            f"Got offset: {offset}, date_offset: {date_offset}"
        )

        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if normalize:
            end = offset if offset is not None else len(
                df[df.index <= date_offset].index) if date_offset is not None else len(df.index)
            df = self.scale(df, 0, end)

        if dataset_type == "long":
            _from_dataframe = _from_long_dataframe
        elif dataset_type == "wide":
            _from_dataframe = _from_wide_dataframe
        elif dataset_type == "wide_multivariate":
            _from_dataframe = _from_wide_dataframe_multivariate
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'long', 'wide', and 'wide_multivariate'."
            )

        example_gen_func, features = _from_dataframe(
            df, freq=freq, offset=offset, date_offset=date_offset
        )
        hf_dataset = datasets.Dataset.from_generator(
            example_gen_func, features=features
        )
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(self.storage_path / self.dataset)

    def load_dataset(
        self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform_map[self.dataset](),
            dataset_weight=self.weight,
            sample_time_series=self.sample_time_series,
        )

    def scale(self, data, start, end):
        train = data[start:end]
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0)
        return (data - self.mean) / (self.std + 1e-10)


@dataclass
class SimpleFinetuneDatasetBuilder(DatasetBuilder):
    dataset: str
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    storage_path: Path = env.CUSTOM_DATA_PATH
    mean = None
    std = None

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        offset: Optional[int] = None,
        date_offset: Optional[pd.Timestamp] = None,
        freq: str = "H",
        normalize: Optional[bool] = False,
    ):

        # QZ: Same as SimpleDatasetBuilder.

        assert offset is None or date_offset is None, (
            "One or neither offset and date_offset must be specified, but not both. "
            f"Got offset: {offset}, date_offset: {date_offset}"
        )

        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if normalize:
            end = offset if offset is not None else len(
                df[df.index <= date_offset].index) if date_offset is not None else len(df.index)
            df = self.scale(df, 0, end)

        if dataset_type == "long":
            _from_dataframe = _from_long_dataframe
        elif dataset_type == "wide":
            _from_dataframe = _from_wide_dataframe
        elif dataset_type == "wide_multivariate":
            _from_dataframe = _from_wide_dataframe_multivariate
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'long', 'wide', and 'wide_multivariate'."
            )

        example_gen_func, features = _from_dataframe(
            df, freq=freq, offset=offset, date_offset=date_offset
        )
        hf_dataset = datasets.Dataset.from_generator(
            example_gen_func, features=features
        )
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(self.storage_path / self.dataset)

    def load_dataset(
        self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        return FinetuneDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform_map[self.dataset](
                distance=self.distance,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
            ),
        )

    def scale(self, data, start, end):
        train = data[start:end]
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0)
        return (data - self.mean) / (self.std + 1e-10)


@dataclass
class SimpleEvalDatasetBuilder(DatasetBuilder):
    dataset: str
    offset: Optional[int]
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        freq: str = "H",
        mean: pd.Series = None,
        std: pd.Series = None,
    ):
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if mean is not None and std is not None:  # Qz: Normalize data like LSF
            df = (df - mean) / (std + 1e-10)

        if dataset_type == "long":
            _from_dataframe = _from_long_dataframe
        elif dataset_type == "wide":
            _from_dataframe = _from_wide_dataframe
        elif dataset_type == "wide_multivariate":
            _from_dataframe = _from_wide_dataframe_multivariate
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'long', 'wide', and 'wide_multivariate'."
            )

        example_gen_func, features = _from_dataframe(df, freq=freq)
        hf_dataset = datasets.Dataset.from_generator(
            example_gen_func, features=features
        )
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(self.storage_path / self.dataset)

    def load_dataset(
        self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        return EvalDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform_map[self.dataset](
                offset=self.offset,
                distance=self.distance,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
            ),
        )


def generate_finetune_builder(
    dataset: str,
    train_length: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    storage_path: Path = env.CUSTOM_DATA_PATH,
) -> SimpleFinetuneDatasetBuilder:
    """
    Set distance=1 for training data. Same as standard LSF setting.
    """
    return SimpleFinetuneDatasetBuilder(
        dataset=dataset,
        windows=train_length - context_length - prediction_length + 1,
        distance=1,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        storage_path=storage_path,
    )


def generate_eval_builder(
    dataset: str,
    offset: int,
    eval_length: int,
    prediction_length: int,
    context_length: int,
    patch_size: int,
    storage_path: Path = env.CUSTOM_DATA_PATH,
) -> SimpleEvalDatasetBuilder:
    """
    Set distance according to dataset. Decrease the number of validation samples to reduce computational cost.
    """

    # distances = {
    #     "ETTh1_eval": 1,  # 13h
    #     "ETTh2_eval": 1,
    #     "ETTm1_eval": 1,  # 6h 15min
    #     "ETTm2_eval": 1,
    #     "weather_eval": 1,  # 6h 10 min
    #     "electricity_eval": 1,  # 2d 1h
    # }

    distances = {
        "ETTh1_eval": 13,  # 13h
        "ETTh2_eval": 13,
        "ETTm1_eval": 25,  # 6h 15min
        "ETTm2_eval": 25,
        "weather_eval": 37,  # 6h 10 min
        "electricity_eval": 49,  # 2d 1h
    }
    if dataset in distances:
        distance = distances[dataset]
        windows = (eval_length - prediction_length) // distance + 1
    else:
        distance = prediction_length
        windows = eval_length // prediction_length

    # base = 8  # base can change for different datasets
    # overlap_ratio = {
    #     96: base,
    #     192: 2*base,
    #     336: 4*base,
    #     720: 8*base,
    # }
    # if prediction_length in overlap_ratio:
    #     distance = prediction_length // overlap_ratio[prediction_length]
    #     windows = (eval_length - prediction_length) // distance + 1
    # else:
    #     distance = prediction_length
    #     windows = eval_length // prediction_length

    return SimpleEvalDatasetBuilder(
        dataset=dataset,
        offset=offset,
        windows=windows,
        distance=distance,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        storage_path=storage_path,
    )


def generate_eval_builders(
    dataset: str,
    offset: int,
    eval_length: int,
    prediction_lengths: list[int],
    context_lengths: list[int],
    patch_sizes: list[int],
    storage_path: Path = env.CUSTOM_DATA_PATH,
) -> list[SimpleEvalDatasetBuilder]:
    return [
        SimpleEvalDatasetBuilder(
            dataset=dataset,
            offset=offset,
            windows=eval_length // pred,
            distance=pred,
            prediction_length=pred,
            context_length=ctx,
            patch_size=psz,
            storage_path=storage_path,
        )
        for pred, ctx, psz in product(prediction_lengths, context_lengths, patch_sizes)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["wide", "long", "wide_multivariate"],
        default="wide",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--date_offset",
        type=str,
        default=None,
    )
    # Define the `freq` argument with a default value. Use this value as 'freq' if 'freq' is None.
    parser.add_argument(
        "--freq",
        default="H",  # Set the default value
        help="The user specified frequency",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize train and eval data with train statistics",
    )

    args = parser.parse_args()

    # Create training dataset
    # If offset/date_offset is not provided, the whole data will be used for training.
    # Otherwise, only the part before offset is used for training.
    train_dataset_builder = SimpleFinetuneDatasetBuilder(
        dataset=args.dataset_name,
        windows=None,
        distance=None,
        prediction_length=None,
        context_length=None,
        patch_size=None,
    )
    train_dataset_builder.build_dataset(
        file=Path(args.file_path),
        dataset_type=args.dataset_type,
        offset=args.offset,
        date_offset=pd.Timestamp(args.date_offset) if args.date_offset else None,
        freq=args.freq,
        normalize=args.normalize,  # Qz: To align with LSF. Default is False
    )

    # Create a validation dataset if offset/date_offset is provided.
    # Eval dataset include the whole data.
    if args.offset is not None or args.date_offset is not None:
        SimpleEvalDatasetBuilder(
            f"{args.dataset_name}_eval",
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
            mean=train_dataset_builder.mean,
            std=train_dataset_builder.std,
        )
