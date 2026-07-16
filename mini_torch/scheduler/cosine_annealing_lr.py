import math

from mini_torch.scheduler.LRScheduler import LRScheduler


class CosineAnnealingLR(LRScheduler):

    def __init__(
        self,
        optimizer,
        total_steps,
        warmup_steps=0,
        eta_min=0.0,
    ):
        super().__init__(optimizer)

        if total_steps <= 0:
            raise ValueError(
                "total_steps must be greater than 0."
            )

        if warmup_steps < 0:
            raise ValueError(
                "warmup_steps cannot be negative."
            )

        if warmup_steps >= total_steps:
            raise ValueError(
                "warmup_steps must be less than total_steps."
            )

        self.base_lr = optimizer.lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min

    def get_lr(self):

        # ==================================================
        # Warmup
        # ==================================================

        if (
            self.warmup_steps > 0
            and self.last_epoch < self.warmup_steps
        ):

            return (
                self.base_lr
                * (self.last_epoch + 1)
                / self.warmup_steps
            )

        # ==================================================
        # Cosine Annealing
        # ==================================================

        cosine_steps = (
            self.total_steps
            - self.warmup_steps
        )

        progress = (
            self.last_epoch
            - self.warmup_steps
        ) / cosine_steps

        # Prevent cosine from restarting if step()
        # is called beyond total_steps.
        progress = min(
            max(progress, 0.0),
            1.0,
        )

        return (
            self.eta_min
            + 0.5
            * (self.base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi * progress
                )
            )
        )

    def state_dict(self):

        state = super().state_dict()

        state.update({
            "base_lr": self.base_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "eta_min": self.eta_min,
        })

        return state

    def load_state_dict(
        self,
        state_dict,
    ):

        self.base_lr = state_dict["base_lr"]

        self.total_steps = state_dict[
            "total_steps"
        ]

        self.warmup_steps = state_dict[
            "warmup_steps"
        ]

        self.eta_min = state_dict[
            "eta_min"
        ]

        super().load_state_dict(
            state_dict
        )