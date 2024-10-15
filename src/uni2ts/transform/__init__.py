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

from ._base import Chain, Identity, Transformation
from .crop import EvalCrop, FinetunePatchCrop, PatchCrop, PatchCropGivenFixedConfig
from .feature import AddObservedMask, AddSampleIndex, AddTimeIndex, AddVariateIndex
from .field import LambdaSetFieldIfNotPresent, RemoveFields, SelectFields, SetValue
from .imputation import DummyValueImputation, ImputeTimeSeries, LastValueImputation
from .multi_scale import (
    AddNewFreqScaleSeries,
    AddNewScaleContextSeries,
    AddNewScaleSeries,
    MultiScaleEvalCrop,
    MultiScaleMaskedPredictionGivenFixedConfig,
    PadNewScaleSeries,
)
from .pad import EvalPad, MaskOutRangePaddedTokens, Pad, PadFreq
from .patch import (
    DefaultPatchSizeConstraints,
    FixedPatchSizeConstraints,
    GetPatchSize,
    Patchify,
    PatchSizeConstraints,
)
from .resample import SampleDimension
from .reshape import (
    FlatPackCollection,
    FlatPackFields,
    PackCollection,
    PackFields,
    SequencifyField,
    Transpose,
)
from .seasonal_naive import (
    AddSeasonalNaiveTarget,
    GetSeasonalNaivePrediction,
    SeasonalNaiveEvalCrop,
)
from .task import (
    EvalMaskedPrediction,
    ExtendMask,
    MaskedPrediction,
    MaskedPredictionGivenFixedConfig,
)

__all__ = [
    "AddNewScaleSeries",
    "AddNewScaleContextSeries",
    "AddNewFreqScaleSeries",
    "AddSampleIndex",
    "AddObservedMask",
    "AddTimeIndex",
    "AddVariateIndex",
    "Chain",
    "DefaultPatchSizeConstraints",
    "DummyValueImputation",
    "EvalCrop",
    "EvalMaskedPrediction",
    "EvalPad",
    "ExtendMask",
    "FixedPatchSizeConstraints",
    "FlatPackCollection",
    "FlatPackFields",
    "GetPatchSize",
    "Identity",
    "ImputeTimeSeries",
    "LambdaSetFieldIfNotPresent",
    "LastValueImputation",
    "MaskedPrediction",
    "MaskedPredictionGivenFixedConfig",
    "MaskOutRangePaddedTokens",
    "MultiScaleEvalCrop",
    "MultiScaleMaskedPredictionGivenFixedConfig",
    "PackCollection",
    "PackFields",
    "Pad",
    "PadFreq",
    "PadNewScaleSeries",
    "PatchCrop",
    "PatchCropGivenFixedConfig",
    "PatchSizeConstraints",
    "Patchify",
    "RemoveFields",
    "SampleDimension",
    "SelectFields",
    "SequencifyField",
    "SetValue",
    "Transformation",
    "Transpose",
    "GetSeasonalNaivePrediction",
    "AddSeasonalNaiveTarget",
    "SeasonalNaiveEvalCrop",
    "FinetunePatchCrop",
]
