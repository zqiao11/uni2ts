import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from einops import pack
from jaxtyping import Bool, Float

from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import CheckArrNDimMixin, MapFuncMixin


@dataclass
class AddNewScaleContextSeries(CheckArrNDimMixin, Transformation):
    """
    Add down-sampled new scales to data_entry. Each scale is downsampled by a factor of powers of 2.
    """

    target_field: str
    num_new_scales_fields: tuple[str, ...]
    expected_ndim: int = 2

    def __post_init__(self):

        self.new_arr_list = []
        self.new_context_length_list = []
        self.context_length_new_scales = {}

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.__post_init__()
        for field in self.num_new_scales_fields:
            data_entry[field] = self._downsample(
                data_entry,
                self.target_field,
            )

        for field in self.num_new_scales_fields:
            self.context_length_new_scales[field] = self.new_context_length_list[
                int(field[-1])
            ]
        data_entry["context_length_new_scales"] = self.context_length_new_scales

        return data_entry

    def _downsample(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        if len(self.new_arr_list) == 0:
            arr = data_entry[field]
        else:
            arr = self.new_arr_list[-1]

        patch_size = data_entry["patch_size"]

        self.check_ndim(field, arr, self.expected_ndim)
        dim, time = arr.shape[:2]
        ds_factor = 2

        if len(self.new_context_length_list) == 0:
            context_length = data_entry["context_length"]
            # Remove padded NAN from target
            context_pad = -context_length % patch_size
            arr = arr[:, context_pad : context_length + context_pad]

        else:
            context_length = self.new_context_length_list[-1]

        # Compute new scale's context after padding.
        new_context_length = math.ceil(context_length / ds_factor)
        context_pad = -context_length % ds_factor

        arr = np.pad(
            arr,
            ((0, 0), (context_pad, 0)),
            mode="constant",
            constant_values=np.nan,
        )  # Pad with NaN

        # Reshape the array to apply pooling (split into non-overlapping windows)
        arr_new = np.nanmean(
            arr.reshape(dim, -1, ds_factor), axis=2
        )  # Compute the mean, ignoring NaN values

        assert (
            arr_new.shape[1] == new_context_length
        ), "Error occurs during downsampling!"

        self.new_arr_list.append(arr_new)
        self.new_context_length_list.append(new_context_length)
        return arr_new


@dataclass
class AddNewScaleSeries(CheckArrNDimMixin, Transformation):
    """
    Add down-sampled new scales to data_entry. Each scale is downsampled by a factor of powers of 2.
    """

    target_field: str
    num_new_scales: int
    expected_ndim: int = 2

    def __post_init__(self):

        self.new_arr_list = []
        self.new_context_length_list = []
        self.new_prediction_length_list = []
        self.context_length_new_scales = {}
        self.prediction_length_new_scales = {}

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.__post_init__()
        for i in range(self.num_new_scales):
            data_entry[f"target{i+1}"] = self._downsample(
                data_entry,
                self.target_field,
            )

        for i in range(self.num_new_scales):
            self.context_length_new_scales[f"target{i+1}"] = (
                self.new_context_length_list[i]
            )
            self.prediction_length_new_scales[f"target{i+1}"] = (
                self.new_prediction_length_list[i]
            )
        data_entry["context_length_new_scales"] = self.context_length_new_scales
        data_entry["prediction_length_new_scales"] = self.prediction_length_new_scales

        return data_entry

    def _downsample(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        if len(self.new_arr_list) == 0:
            arr = data_entry[field]
        else:
            arr = self.new_arr_list[-1]

        patch_size = data_entry["patch_size"]

        self.check_ndim(field, arr, self.expected_ndim)
        dim, time = arr.shape[:2]
        ds_factor = 2

        if len(self.new_context_length_list) == 0:
            context_length = data_entry["context_length"]
            prediction_length = data_entry["prediction_length"]

            # Remove padded NAN from target
            context_pad = -context_length % patch_size
            prediction_pad = -prediction_length % patch_size
            if prediction_pad > 0:
                arr = arr[:, context_pad:-prediction_pad]
            else:
                arr = arr[:, context_pad:]

        else:
            context_length = self.new_context_length_list[-1]
            prediction_length = self.new_prediction_length_list[-1]

        # Compute new scale's context and prediction length after padding.
        new_context_length = math.ceil(context_length / ds_factor)
        new_prediction_length = math.ceil(prediction_length / ds_factor)

        # Pad context and prediction separately
        context_pad = -context_length % ds_factor
        prediction_pad = -prediction_length % ds_factor

        arr = np.pad(
            arr,
            ((0, 0), (context_pad, prediction_pad)),
            mode="constant",
            constant_values=np.nan,
        )  # Pad with NaN

        # Reshape the array to apply pooling (split into non-overlapping windows)
        arr_new = np.nanmean(
            arr.reshape(dim, -1, ds_factor), axis=2
        )  # Compute the mean, ignoring NaN values

        assert (
            arr_new.shape[1] == new_context_length + new_prediction_length
        ), "Error occurs during downsampling!"

        self.new_arr_list.append(arr_new)
        self.new_context_length_list.append(new_context_length)
        self.new_prediction_length_list.append(new_prediction_length)

        return arr_new


@dataclass
class AddNewFreqScaleSeries(CheckArrNDimMixin, Transformation):
    """
    Add down-sampled new scales to data_entry based on its freq
    """

    fields: tuple[str, ...]
    num_new_scales: int
    optional_fields: tuple[str, ...] = tuple()
    expected_ndim: int = 2
    collection_type: type = list

    DOWN_SAMPLE_FACTOR = {
        "S": 60,  # Seconds to Minutes
        "T": 60,  # Minutes to Hours
        "H": 24,  # Hours to Days
        "D": 7,  # Days to Weeks
        "W": 4,  # Weeks to Months
        "M": 4,  # Months to Quarters
        "Q": 3,  # Quarters to Years
    }

    FREQ_ORDER = ["S", "T", "H", "D", "W", "M", "Q"]

    def __post_init__(self):
        self.freq_per_scale = []
        self.new_arr_list = []
        self.new_context_length_list = []
        self.new_prediction_length_list = []
        # self.freq_new_scales = {}
        self.context_length_new_scales = {}
        self.prediction_length_new_scales = {}

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.__post_init__()
        self.freq_per_scale.append(data_entry["freq"])
        for i in range(self.num_new_scales):
            data_entry[f"target{i+1}"] = self._downsample(
                data_entry,
                self.fields,
            )

        # Save the freq for new scales
        for i in range(self.num_new_scales):
            self.context_length_new_scales[f"target{i+1}"] = (
                self.new_context_length_list[i]
            )
            self.prediction_length_new_scales[f"target{i+1}"] = (
                self.new_prediction_length_list[i]
            )
        data_entry["context_length_new_scales"] = self.context_length_new_scales
        data_entry["prediction_length_new_scales"] = self.prediction_length_new_scales

        return data_entry

    def _downsample(self, data_entry: dict[str, Any], field: str) -> np.ndarray:
        if len(self.new_arr_list) == 0:
            arr = data_entry[field]
        else:
            arr = self.new_arr_list[-1]

        patch_size = data_entry["patch_size"]
        freq = self.freq_per_scale[-1]
        self.check_ndim(field, arr, self.expected_ndim)
        dim, time = arr.shape[:2]
        # ds_factor = self.DOWN_SAMPLE_FACTOR[freq]
        ds_factor, freq_unit = self.get_downsample_factor(freq)

        if len(self.new_context_length_list) == 0:
            context_length = data_entry["context_length"]
            prediction_length = data_entry["prediction_length"]

            # Remove padded NAN from target
            context_pad = -context_length % patch_size
            prediction_pad = -prediction_length % patch_size
            if prediction_pad > 0:
                arr = arr[:, context_pad:-prediction_pad]
            else:
                arr = arr[:, context_pad:]

        else:
            context_length = self.new_context_length_list[-1]
            prediction_length = self.new_prediction_length_list[-1]

        # Compute new scale's context and prediction length after padding.
        new_context_length = math.ceil(context_length / ds_factor)
        new_prediction_length = math.ceil(prediction_length / ds_factor)

        # Pad context and prediction separately
        context_pad = -context_length % ds_factor
        prediction_pad = -prediction_length % ds_factor

        arr = np.pad(
            arr,
            ((0, 0), (context_pad, prediction_pad)),
            mode="constant",
            constant_values=np.nan,
        )  # Pad with NaN

        # Reshape the array to apply pooling (split into non-overlapping windows)
        arr_new = np.nanmean(
            arr.reshape(dim, -1, ds_factor), axis=2
        )  # Compute the mean, ignoring NaN values

        assert (
            arr_new.shape[1] == new_context_length + new_prediction_length
        ), "Error occurs during downsampling!"

        self.new_arr_list.append(arr_new)
        self.freq_per_scale.append(self.get_next_freq(freq_unit))
        self.new_context_length_list.append(new_context_length)
        self.new_prediction_length_list.append(new_prediction_length)

        return arr_new

    def get_next_freq(self, current_freq: str) -> str:
        idx = self.FREQ_ORDER.index(current_freq)
        if idx + 1 < len(self.FREQ_ORDER):
            return self.FREQ_ORDER[idx + 1]
        return current_freq

    def get_downsample_factor(self, freq: str) -> int:
        # Regex pattern to capture numeric prefix and frequency unit (e.g., "10T")
        match = re.match(r"(\d+)?([A-Z]+)", freq)

        if match:
            num_part, freq_unit = match.groups()

            # Retrieve the down-sample factor for the frequency unit (e.g., "T", "H", etc.)
            ds_factor = self.DOWN_SAMPLE_FACTOR[freq_unit]

            # If there's a numeric prefix, adjust the factor by dividing it
            if num_part is not None:
                ds_factor = ds_factor // int(
                    num_part
                )  # Adjust by dividing the original factor

            return ds_factor, freq_unit
        else:
            raise ValueError(f"Invalid frequency format: {freq}")


@dataclass
class PadNewScaleSeries(MapFuncMixin, Transformation):
    """
    Pad the new scale series for patching. Make their lengths divisible by patch_size.
    """

    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        self.map_func(
            self.map,
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    def map(self, data_entry: dict[str, Any], field: str) -> Any:
        patch_size = data_entry["patch_size"]
        arr = data_entry[field]
        context_length = data_entry["context_length_new_scales"][field]
        context_pad = -context_length % patch_size

        if "prediction_length_new_scales" in data_entry:
            prediction_length = data_entry["prediction_length_new_scales"][field]
            prediction_pad = -prediction_length % patch_size
        else:
            prediction_pad = 0

        pad_width = [(0, 0) for _ in range(arr.ndim)]
        pad_width[-1] = (context_pad, prediction_pad)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=np.nan)
        return arr


@dataclass
class MultiScaleMaskedPredictionGivenFixedConfig(CheckArrNDimMixin, Transformation):
    target_fields: tuple[str, ...] = ("target",)
    prediction_mask_field: str = "prediction_mask"
    expected_ndim: int = 2

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        prediction_mask_dict = {}

        for field in self.target_fields:
            target = data_entry[field]

            if field == "target":
                mask_length = data_entry["num_pred_patches"]
                data_entry["prediction_length"]
            else:
                if "prediction_length_new_scales" in data_entry:
                    mask_length = math.ceil(
                        data_entry["prediction_length_new_scales"][field]
                        / data_entry["patch_size"]
                    )
                else:
                    mask_length = 0

            prediction_mask = self._generate_prediction_mask(target, mask_length)
            prediction_mask_dict[field] = prediction_mask

        data_entry[self.prediction_mask_field] = prediction_mask_dict
        return data_entry

    def _generate_prediction_mask(
        self, target: Float[np.ndarray, "var time *feat"], mask_length: int
    ) -> Bool[np.ndarray, "var time"]:
        self.check_ndim("target", target, self.expected_ndim)
        var, time = target.shape[:2]
        prediction_mask = np.zeros((var, time), dtype=bool)
        if mask_length > 0:
            prediction_mask[:, -mask_length:] = True
        return prediction_mask


@dataclass
class MultiScaleEvalCrop(MapFuncMixin, Transformation):
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
