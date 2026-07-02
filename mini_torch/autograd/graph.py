from mini_torch.tensors import tensor


def topological_sort(root: tensor) -> list[tensor]:
    """
    Return a topologically sorted list of tensors in the computational graph.

    The returned list is ordered from leaf tensors to the root tensor.
    """

    visited = set()
    order = []

    def dfs(node: tensor):
        # Skip nodes we've already processed.
        if node in visited:
            return

        visited.add(node)

        # Visit all dependencies first.
        for parent in node.parents:
            dfs(parent)

        # Record the node after its parents.
        order.append(node)

    dfs(root)

    return order
