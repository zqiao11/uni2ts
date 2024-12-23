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

from typing import List, NamedTuple, Optional, cast
from .simple import _from_long_dataframe, _from_wide_dataframe, _from_wide_dataframe_multivariate

import numpy as np

# Codes are adopted from Gluonts dataset module
# https://ts.gluon.ai/stable/_modules/gluonts/dataset/repository/datasets.html
# https://github.com/awslabs/gluonts/blob/dev/src/gluonts/dataset/repository/_lstnet.py

root = (
    "https://raw.githubusercontent.com/laiguokun/"
    "multivariate-time-series-data/master/"
)


class LstnetDataset(NamedTuple):
    name: str
    url: str
    num_series: int
    num_time_steps: int
    prediction_length: int
    rolling_evaluations: int
    freq: str
    start_date: str
    agg_freq: Optional[str] = None


datasets_info = {
    "exchange_rate": LstnetDataset(
        name="exchange_rate",
        url=root + "exchange_rate/exchange_rate.txt.gz",
        num_series=8,
        num_time_steps=7588,
        prediction_length=30,
        rolling_evaluations=5,
        start_date="1990-01-01",
        freq="1B",
        agg_freq=None,
    ),
    "electricity": LstnetDataset(
        name="electricity",
        url=root + "electricity/electricity.txt.gz",
        # original dataset can be found at
        # https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014#
        # the aggregated ones that is used from LSTNet filters out from the
        # initial 370 series the one with no data in 2011
        num_series=321,
        num_time_steps=26304,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2012-01-01",
        freq="1h",
        agg_freq=None,
    ),
    "traffic": LstnetDataset(
        name="traffic",
        url=root + "traffic/traffic.txt.gz",
        # note there are 963 in the original dataset from
        # https://archive.ics.uci.edu/ml/datasets/PEMS-SF but only 862 in
        # LSTNet
        num_series=862,
        num_time_steps=17544,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2015-01-01",
        freq="h",
        agg_freq=None,
    ),
    "solar-energy": LstnetDataset(
        name="solar-energy",
        url=root + "solar-energy/solar_AL.txt.gz",
        num_series=137,
        num_time_steps=52560,
        prediction_length=24,
        rolling_evaluations=7,
        start_date="2006-01-01",
        freq="10min",
        agg_freq="1h",  # ToDo: Solar原本freq是10min  agg_freq后变成1h
    ),
}


def load_from_pandas(
    df: pd.DataFrame,
    time_index: pd.DatetimeIndex,
    agg_freq: Optional[str] = None,
) -> List[pd.Series]:
    df: pd.DataFrame = df.set_index(time_index)

    pivot_df = df.transpose()
    pivot_df.head()

    timeseries = []
    for row in pivot_df.iterrows():
        ts = pd.Series(row[1].values, index=time_index)
        if agg_freq is not None:
            ts = ts.resample(agg_freq).sum()
        first_valid = ts[ts.notnull()].index[0]
        last_valid = ts[ts.notnull()].index[-1]
        ts = ts[first_valid:last_valid]

        timeseries.append(ts)

    return timeseries


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

        assert offset is None or date_offset is None, (
            "One or neither offset and date_offset must be specified, but not both. "
            f"Got offset: {offset}, date_offset: {date_offset}"
        )

        # gluonts datasets
        if self.dataset in datasets_info:
            ds_info = datasets_info[self.dataset]
            time_index = pd.period_range(
                start=ds_info.start_date,
                freq=ds_info.freq,
                periods=ds_info.num_time_steps,
            )
            time_index = time_index.to_timestamp()

            df = cast(
                pd.DataFrame,
                pd.read_csv(ds_info.url, header=None),  # type: ignore
            )
            timeseries = load_from_pandas(
                df=df, time_index=time_index, agg_freq=ds_info.agg_freq
            )
            df = pd.concat(timeseries, axis=1)

        # gluonts datasets
        else:
            if 'istanbul_traffic' in self.dataset:
                df = pd.read_csv(file, parse_dates=['datetime'], index_col='datetime')
                df = df.resample("h").mean()
            elif 'turkey_power' in self.dataset:
                df = pd.read_csv(file)
                df.Date_Time = pd.to_datetime(df.Date_Time, format="%d.%m.%Y %H:%M")
                df = df.set_index("Date_Time")

            elif 'walmart' in self.dataset:
                df = pd.read_csv(file)
                data = []
                for id, row in df[["Store", "Dept"]].drop_duplicates().iterrows():
                    row_df = df.query(f"Store == {row.Store} and Dept == {row.Dept}")
                    if len(row_df) != 143:
                        continue
                    data.append(row_df.Weekly_Sales.to_numpy())
                data = np.stack(data, 1)
                column_names = range(data.shape[1])
                df = pd.DataFrame(data, columns=column_names)
                df.index = pd.date_range(start="2010-02-05", periods=data.shape[0], freq="W")

            else:
                df = pd.read_csv(file, index_col=0, parse_dates=True)

        if normalize:
            end = (
                offset
                if offset is not None
                else (
                    len(df[df.index <= date_offset].index)
                    if date_offset is not None
                    else len(df.index)
                )
            )
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
        hf_dataset.save_to_disk(self.storage_path / 'pf' / self.dataset)

    def load_dataset(
        self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        return FinetuneDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / 'pf' / self.dataset),
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
        # gluonts datasets
        if self.dataset.removesuffix("_eval") in datasets_info:
            ds_info = datasets_info[self.dataset.removesuffix("_eval")]

            time_index = pd.period_range(
                start=ds_info.start_date,
                freq=ds_info.freq,
                periods=ds_info.num_time_steps,
            )
            time_index = time_index.to_timestamp()

            df = cast(
                pd.DataFrame,
                pd.read_csv(ds_info.url, header=None),  # type: ignore
            )
            timeseries = load_from_pandas(
                df=df, time_index=time_index, agg_freq=ds_info.agg_freq
            )
            df = pd.concat(timeseries, axis=1)

        # gluonts datasets
        else:
            if 'istanbul_traffic' in self.dataset:
                df = pd.read_csv(file, parse_dates=['datetime'], index_col='datetime')
                df = df.resample("h").mean()
            elif 'turkey_power' in self.dataset:
                df = pd.read_csv(file)
                df.Date_Time = pd.to_datetime(df.Date_Time, format="%d.%m.%Y %H:%M")
                df = df.set_index("Date_Time")
            elif 'walmart' in self.dataset:
                df = pd.read_csv(file)
                data = []
                for id, row in df[["Store", "Dept"]].drop_duplicates().iterrows():
                    row_df = df.query(f"Store == {row.Store} and Dept == {row.Dept}")
                    if len(row_df) != 143:
                        continue
                    data.append(row_df.Weekly_Sales.to_numpy())
                data = np.stack(data, 1)
                column_names = range(data.shape[1])
                df = pd.DataFrame(data, columns=column_names)
                df.index = pd.date_range(start="2010-02-05", periods=data.shape[0], freq="W")
            else:
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
        hf_dataset.save_to_disk(self.storage_path / 'pf' / self.dataset)

    def load_dataset(
        self, transform_map: dict[str, Callable[..., Transformation]]
    ) -> Dataset:
        return EvalDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / 'pf' / self.dataset),
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

    return SimpleEvalDatasetBuilder(
        dataset=dataset,
        offset=offset,
        windows=eval_length,
        distance=1,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        storage_path=storage_path,
    )


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
