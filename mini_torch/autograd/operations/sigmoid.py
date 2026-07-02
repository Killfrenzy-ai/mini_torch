from mini_torch.autograd.operation import Operation


class Sigmoid(Operation):

    def backward(self, node, grad_output):

        output = node.data

        grad_input = grad_output * output * (1 - output)

        return (grad_input,)

SIGMOID = Sigmoid()