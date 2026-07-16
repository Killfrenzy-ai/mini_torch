from mini_torch.optim.optimizer import Optimizer
from mini_torch.backend import xp

class AdamW(Optimizer):

    def __init__(self, parameters, lr= 1e-3, beta1= 0.9, beta2 = 0.999, eps= 1e-8, weight_decay= 0.01):

        super().__init__(parameters)

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay

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

    def state_dict(self):

        state = super().state_dict()

        state.update({

            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "t": self.t,
            })

        return state

    def load_state_dict(self, state_dict):

        super().load_state_dict(state_dict)

        self.lr = state_dict["lr"]
        self.beta1 = state_dict["beta1"]
        self.beta2 = state_dict["beta2"]
        self.eps = state_dict["eps"]
        self.t = state_dict["t"]

    def step(self):
        """
        Perform one AdamW optimization step.
        """

        self.t += 1

        for parameter in self.parameters:

            if parameter.grad is None:
                continue

            grad = parameter.grad

            if self.weight_decay > 0:
                grad = grad.copy()

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
            parameter.data -= (self.lr* m_hat/ (xp().sqrt(v_hat) + self.eps))

            if self.weight_decay > 0:

                parameter.data *= (1.0 - self.lr * self.weight_decay)