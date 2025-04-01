from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts import maybe
from gluonts.model import Forecast


def plot_single(
    inp: dict,
    label: dict,
    forecast: Forecast,
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    ax: Optional[plt.axis] = None,
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    ax = maybe.unwrap_or_else(ax, plt.gca)

    target = np.concatenate([inp["target"], label["target"]], axis=-1)
    start = inp["start"]
    if dim is not None:
        target = target[dim]
        forecast = forecast.copy_dim(dim)

    index = pd.period_range(start, periods=len(target), freq=start.freq)
    ax.plot(
        index.to_timestamp()[-context_length - forecast.prediction_length :],
        target[-context_length - forecast.prediction_length :],
        label="target",
        color="black",
    )
    forecast.plot(
        intervals=intervals,
        ax=ax,
        color="blue",
        name=name,
        show_label=show_label,
    )
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left")


def plot_next_multi(
    axes: np.ndarray,
    input_it: Iterator[dict],
    label_it: Iterator[dict],
    forecast_it: Iterator[Forecast],
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for ax, inp, label, forecast in zip(axes, input_it, label_it, forecast_it):
        plot_single(
            inp,
            label,
            forecast,
            context_length,
            intervals=intervals,
            ax=ax,
            dim=dim,
            name=name,
            show_label=show_label,
        )


# def plot_single(
#     inp: dict,
#     label: dict,
#     forecast: Forecast,
#     context_length: int,
#     intervals: tuple[float, ...] = (0.5, 0.9),
#     ax: Optional[plt.axis] = None,
#     dim: Optional[int] = None,
#     name: Optional[str] = None,
#     show_label: bool = False,
# ):
#     ax = maybe.unwrap_or_else(ax, plt.gca)
#
#     target = np.concatenate([inp["target"], label["target"]], axis=-1)
#     start = inp["start"]
#     if dim is not None:
#         target = target[dim]
#         forecast = forecast.copy_dim(dim)
#
#     index = pd.period_range(start, periods=len(target), freq=start.freq)
#     ax.plot(
#         index.to_timestamp()[-context_length - forecast.prediction_length :],
#         target[-context_length - forecast.prediction_length :],
#         label="target",
#         color="black",
#     )
#     forecast.plot(
#         intervals=intervals,
#         ax=ax,
#         color="blue",
#         name=name,
#         show_label=show_label,
#     )
#     # Modify the x-axis tick font size
#     ax.tick_params(axis='x', labelsize=8)
#
#     # Modify the y-axis tick font size
#     ax.tick_params(axis='y', labelsize=15)
#     ax.set_xticks(ax.get_xticks())
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#     ax.legend(loc="lower left", fontsize=25)
#
#     # ax.set_ylim(0.8, 2.6)
#
# def plot_next_multi(
#     axes: np.ndarray,
#     input_it: Iterator[dict],
#     label_it: Iterator[dict],
#     forecast_it: Iterator[Forecast],
#     context_length: int,
#     intervals: tuple[float, ...] = (0.5, 0.9),
#     dim: Optional[int] = None,
#     name: Optional[str] = None,
#     show_label: bool = False,
# ):
#     axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
#     counter=0
#     for ax, inp, label, forecast in zip(axes, input_it, label_it, forecast_it):
#         counter += 1
#         steps = 500 if counter != 4 else 1000
#         for _ in range(steps):
#             next(input_it)
#             next(label_it)
#             next(forecast_it)
#
#         plot_single(
#             inp,
#             label,
#             forecast,
#             context_length,
#             intervals=intervals,
#             ax=ax,
#             dim=dim,
#             name=name,
#             show_label=show_label,
#         )