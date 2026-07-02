from mini_torch.autograd.operation import Operation

class ReLU(Operation):
    """Backward rule for ReLU activation function."""

    def backward(self, node, grad_output):
        parent, = node.parents

        grad = grad_output * (parent.data > 0)

        return (grad,)
    
RELU = ReLU()