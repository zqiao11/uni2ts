import logging
from typing import Optional, Tuple

import torch
from lightning.pytorch.callbacks import EarlyStopping
from torch import Tensor

log = logging.getLogger(__name__)


class WarmupEarlyStopping(EarlyStopping):
    def __init__(self, warmup_steps: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)

        self.warmup_steps = warmup_steps
        self.warmup_finished = False

    def _evaluate_stopping_criteria(
        self, current: Tensor
    ) -> Tuple[bool, Optional[str]]:
        should_stop = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(
            current, self.stopping_threshold
        ):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(
            -current, -self.divergence_threshold
        ):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(
            current - self.min_delta, self.best_score.to(current.device)
        ):
            should_stop = False
            reason = self._improvement_message(current)

            if self.warmup_steps is not None:
                if self.wait_count < self.warmup_steps and not self.warmup_finished:
                    should_stop = False
                    reason = f"The first {self.warmup_steps} improved validations are for warmup, not included for earlystop monitoring."
                    self.wait_count += 1

                    if self.wait_count >= self.warmup_steps:
                        self.warmup_finished = True
                else:
                    self.best_score = current
                    self.wait_count = 0

            else:
                self.best_score = current
                self.wait_count = 0
        else:
            self.wait_count += 1

            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                    f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                )

        return should_stop, reason
