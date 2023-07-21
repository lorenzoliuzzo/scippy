from .topology import find_topo_sort

def gradients(loss_node, nodes):
    """Compute gradients of nodes with respect to the loss node using backpropagation.

    Parameters
    ----------
    loss_node: Node
        The output node (scalar) representing the loss.
    nodes: List[Node]
        List of input nodes with respect to which the gradients are computed.

    Returns
    -------
    gradients: Dict[Node, Value]
        Dictionary mapping input nodes to their corresponding gradients.
    """
    # Initialize gradients dictionary with the loss_node's gradient
    gradients = {loss_node: 1.0}

    # Find the topological sort of nodes ending in the loss node
    topo_order = find_topo_sort([loss_node])

    # Perform reverse-mode automatic differentiation (backpropagation)
    for node in reversed(topo_order):
        # Get the gradient of the current node with respect to its output
        node_grad = gradients[node]

        # Get the gradients of the node with respect to its inputs
        input_gradients = node.op.gradient(node, node_grad)

        # Update gradients for input nodes of the current node
        for i, input_node in enumerate(node.inputs):
            if gradients.get(input_node) is None:
                gradients[input_node] = input_gradients[i]
            else:
                gradients[input_node] += input_gradients[i]

    # Collect gradients for the specified input
    input_gradients = {node: gradients.get(node, 0.0) for node in nodes}

    return input_gradients
