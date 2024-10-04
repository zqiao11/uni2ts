import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from einops import pack, rearrange
from jaxtyping import Bool, Float, Num

from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import (
    AddNewArrMixin,
    ApplyFuncMixin,
    CheckArrNDimMixin,
    CollectFuncMixin,
    MapFuncMixin,
)


def seasonal_naive_predict(context: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Apply seasonal naive prediction to forecast the prediction time series based on the context.

    Args:
        context (np.ndarray): Time series of shape (feat, context_len) that provides the historical data.
        prediction (np.ndarray): Time series of shape (feat, prediction_len) to be predicted using seasonal naive method.

    Returns:
        np.ndarray: Forecasted time series of shape (feat, prediction_len) using seasonal naive method.
    """

    # Get the number of features and lengths for both context and prediction
    feat, context_len = context.shape
    _, prediction_len = prediction.shape

    # Initialize the forecast array with the same shape as the prediction array
    forecast = np.zeros_like(prediction)

    # Iterate through each feature separately
    for i in range(feat):
        # Compute FFT on the context to find the dominant period
        fft_vals = np.fft.fft(context[i])
        freqs = np.fft.fftfreq(context_len)

        # Discard the freq=0 component by starting from index 1
        fft_vals = fft_vals[1:]
        freqs = freqs[1:]

        # Identify the period by finding the frequency with the highest power
        dominant_freq = freqs[np.argmax(np.abs(fft_vals))]

        # Compute the period length from the dominant frequency
        period = int(np.abs(1 / dominant_freq))

        # ToDo: For now, we only consider the case that context is longer than prediction
        # If no periodicity in context, use the last time points for forecasting.
        if period == context_len:
            forecast = context[i, -prediction_len:]
        else:
            # Apply the seasonal naive method to forecast
            for t in range(prediction_len):
                # Forecast based on repeating the seasonal pattern
                forecast[i, t] = context[
                    i, (context_len - period + (t % period)) % context_len
                ]

    return forecast


@dataclass
class GetSeasonalNaivePrediction(Transformation):
    """
    Forecast the prediction range with SeasonalityNaive
    """

    naive_prediction_field: str

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        """
        target: ndarray of shape (feat, time). It has been padded to make sure it can be divided by patch_size.
        """
        target = data_entry["target"]
        context_length = data_entry["context_length"]
        prediction_length = data_entry["prediction_length"]
        patch_size = data_entry["patch_size"]

        context_pad = -context_length % patch_size
        prediction_pad = -prediction_length % patch_size

        context = target[:, context_pad : context_pad + context_length]

        if prediction_pad == 0:
            prediction = target[:, -prediction_length:]
        else:
            prediction = target[
                :, -(prediction_pad + prediction_length) : -prediction_pad
            ]

        season_naive_prediction = seasonal_naive_predict(context, prediction)

        data_entry[self.naive_prediction_field] = season_naive_prediction

        return data_entry


@dataclass
class AddSeasonalNaiveTarget(Transformation):
    max_patch_size: int
    naive_target_field: str
    naive_prediction_field: str
    pad_value: int | float = 0

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        target = data_entry["target"]
        patch_size = data_entry["patch_size"]
        num_pred_patches = data_entry["num_pred_patches"]
        prediction_length = data_entry["prediction_length"]
        season_naive_prediction = data_entry[self.naive_prediction_field]

        prediction_pad = -prediction_length % patch_size

        # Pad with zeros to get prediction patches
        pad_width = [(0, 0) for _ in range(season_naive_prediction.ndim)]
        pad_width[-1] = (0, prediction_pad)
        season_naive_prediction = np.pad(
            season_naive_prediction, pad_width, mode="constant", constant_values=0
        )

        season_naive_prediction_patches = self._patchify_arr(
            season_naive_prediction, patch_size
        )
        target[:, -num_pred_patches:, :] = season_naive_prediction_patches
        data_entry[self.naive_target_field] = target

        return data_entry

    def _patchify_arr(
        self, arr: Num[np.ndarray, "var time*patch"], patch_size: int
    ) -> Num[np.ndarray, "var time max_patch"]:
        assert arr.shape[-1] % patch_size == 0
        arr = rearrange(arr, "... (time patch) -> ... time patch", patch=patch_size)
        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (0, self.max_patch_size - patch_size)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=self.pad_value)
        return arr


@dataclass
class SeasonalNaiveEvalCrop(MapFuncMixin, Transformation):
    offset: int
    distance: int
    prediction_length: int
    context_length: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        data_entry["context_length"] = self.context_length
        data_entry["prediction_length"] = self.prediction_length
        data_entry["num_pred_patches"] = math.ceil(
            self.prediction_length / data_entry["patch_size"]
        )
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a : b or None] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]
        window = data_entry["window"]
        fcst_start = self.offset + window * self.distance
        a = fcst_start - self.context_length
        b = fcst_start + self.prediction_length

        if self.offset >= 0:
            assert time >= b > a >= 0
        else:
            assert 0 >= b > a >= -time

        return a, b
