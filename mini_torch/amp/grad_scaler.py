from mini_torch.backend import xp


class GradScaler:

    def __init__(
        self,
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
    ):

        self.scale = init_scale

        self.growth_factor = (
            growth_factor
        )

        self.backoff_factor = (
            backoff_factor
        )

        self.growth_interval = (
            growth_interval
        )

        self._growth_tracker = 0


    def scale_loss(self, loss):

        return loss * self.scale


    def unscale_(self, optimizer):

        inv_scale = (
            1.0 / self.scale
        )

        for parameter in optimizer.parameters:

            if parameter.grad is not None:

                parameter.grad *= (
                    inv_scale
                )


    def found_inf(self, optimizer):

        for parameter in optimizer.parameters:

            if parameter.grad is None:
                continue

            if not xp().all(
                xp().isfinite(
                    parameter.grad
                )
            ):
                return True

        return False


    def update(self, found_inf):

        if found_inf:

            self.scale *= (
                self.backoff_factor
            )

            self._growth_tracker = 0

        else:

            self._growth_tracker += 1

            if (
                self._growth_tracker
                >= self.growth_interval
            ):

                self.scale *= (
                    self.growth_factor
                )

                self._growth_tracker = 0