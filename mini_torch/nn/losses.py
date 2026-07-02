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