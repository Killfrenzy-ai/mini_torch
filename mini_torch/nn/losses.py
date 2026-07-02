from mini_torch.nn.module import Module

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
        - prediction: probabilities after Softmax
        - target: one-hot encoded labels

    Example:
        prediction = [[0.1, 0.8, 0.1]]
        target     = [[0.0, 1.0, 0.0]]
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):

        prediction = prediction.clip(
            self.eps,
            1.0 - self.eps,
        )

        loss = -(target * prediction.log())

        return loss.sum(axis=1).mean()