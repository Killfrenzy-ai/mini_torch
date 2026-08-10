from mini_torch.nn.module import Module
from mini_torch.backend import xp
from mini_torch.tensors import tensor

class MSELoss(Module):
    """
    Mean Squared Error Loss.

    This loss function computes the mean squared error between the predicted and target values.
    """

    def forward(self, prediction, target):
        """
        Forward pass of the MSE loss.

        Args:
            prediction (tensor): Predicted values.
            target (tensor): Ground truth values.

        Returns:
            tensor: Computed MSE loss.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} "
                f"does not match target shape {target.shape}."
    )
        return ((prediction - target) ** 2).mean()
    
class BCELoss(Module):
    """
    Binary Cross Entropy Loss.

    Expects predictions to already be passed through a Sigmoid.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):

        prediction = prediction.clip(
            self.eps,
            1.0 - self.eps,
        )

        loss = -(
            target * prediction.log()
            +
            (1 - target) * (1 - prediction).log()
        )

        return loss.mean()
    
class CrossEntropyLoss(Module):
    """
    Multi-class Cross Entropy Loss.

    Expects:
        prediction : probabilities
                     Shape (N, C)

        target : integer class labels
                 Shape (N,)
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        Numerically stable Cross Entropy Loss.

        Parameters
        ----------
        logits
            Shape (N, C)

        targets
            Shape (N,)
        """

        # ------------------------------------------
        # Numerical stability
        # ------------------------------------------

        row_max = logits.max(
            axis=-1,
            keepdims=True,
        )

        shifted = logits - row_max

        # ------------------------------------------
        # log(sum(exp(logits)))
        # ------------------------------------------

        logsumexp = (

            shifted.exp()

            .sum(axis=-1)

            .log()

            +

            row_max.squeeze(-1)

        )

        # ------------------------------------------
        # Gather correct class logits
        # ------------------------------------------

        batch_size = logits.shape[0]

        target_logits = logits[
            range(batch_size),
            targets,
        ]

        # ------------------------------------------
        # Cross entropy
        # ------------------------------------------

        loss = (logsumexp - target_logits).mean()

        out = tensor(loss.data, parents=(logits,), op = CROSS_ENTROPY, requires_grad = logits.requires_grad)

        out.logits = logits
        out.targets = targets

        return out