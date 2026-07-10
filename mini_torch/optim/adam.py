from mini_torch.backend import xp
from mini_torch.optim.optimizer import Optimizer

class Adam(Optimizer):
    """
    Adam optimizer.
    """

    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.t = 0

        self._initialize_state()

    def _initialize_state(self):
        """
        Initializes the optimizer state for each parameter.
        """
        for param in self.parameters:
            self.state[id(param)] = {
                "m": xp().zeros_like(param.data),
                "v": xp().zeros_like(param.data),
            }

    def step(self):
        """
        Perform one Adam optimization step.
        """

        self.t += 1

        for parameter in self.parameters:

            if parameter.grad is None:
                continue

            grad = parameter.grad

            state = self.state[id(parameter)]

            m = state["m"]
            v = state["v"]

            # -----------------------------
            # Update biased first moment
            # -----------------------------
            m *= self.beta1
            m += (1.0 - self.beta1) * grad

            # -----------------------------
            # Update biased second moment
            # -----------------------------
            v *= self.beta2
            v += (1.0 - self.beta2) * (grad ** 2)

            # -----------------------------
            # Bias correction
            # -----------------------------
            m_hat = m / (1.0 - self.beta1 ** self.t)

            v_hat = v / (1.0 - self.beta2 ** self.t)

            # -----------------------------
            # Parameter update
            # -----------------------------
            parameter.data -= (
                self.lr
                * m_hat
                / (xp().sqrt(v_hat) + self.eps)
            )
