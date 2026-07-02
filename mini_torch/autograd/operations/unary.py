from mini_torch.autograd.operation import Operation

class Neg(Operation):
    """Backward rule for negation."""

    def backward(self, node, grad_output):
        return (-grad_output,)
    
class Exp(Operation):
    """Backward rule for exponential."""

    def backward(self, node, grad_output):

        grad = grad_output * node.data

        return (grad,)
    
class Log(Operation):
    """Backward rule for natural logarithm."""

    def backward(self, node, grad_output):

        parent, = node.parents

        grad = grad_output / parent.data

        return (grad,)