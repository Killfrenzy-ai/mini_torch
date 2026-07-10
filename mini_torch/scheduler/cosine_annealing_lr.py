import math
from mini_torch.scheduler.LRScheduler import LRScheduler

class CosineAnnealingLR(LRScheduler):

    def __init__(
        self,
        optimizer,
        T_max,
        eta_min=0.0,
    ):
        super().__init__(optimizer)

        self.base_lr = optimizer.lr
        self.T_max = T_max
        self.eta_min = eta_min

    def get_lr(self):

        return (
            self.eta_min
            +
            0.5
            * (self.base_lr - self.eta_min)
            * (
                1
                + math.cos(
                    math.pi
                    * self.last_epoch
                    / self.T_max
                )
            )
        )