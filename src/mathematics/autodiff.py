import numpy as np

# Node in a computation graph.
class Node(object):

    def __init__(self, name=None, value=None):
        """Constructor, new node is indirectly created by Op object __call__ method.
        
        Instance variables:
            self.name: node name for debugging purposes.
            self.value: value of the node computed during forward pass
            self.grad: gradient of the node computed during backward pass
            self.inputs: the list of input nodes.
            self.op: the associated op object, e.g. add_op object if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant, e.g. self.const_attr=5 if this node is created by x+5.
        """
        self.name = name
        self.value = value
        self.grad = None
        self.inputs = []
        self.op = None
        self.const_attr = None


    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in the new node's const_attr field.
            new_node = add_byconst_op(self, other)
        return new_node
    

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Multiply by a constant stores the constant in the new node's const_attr field.
            new_node = mul_byconst_op(self, other)
        return new_node
    

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__


    def __matmul__(self, other):
        """Matrix multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = matmul_op(self, other)
        else:
            # Matrix multiply by a constant stores the constant in the new node's const_attr field.
            new_node = matmul_byconst_op(self, other)
        return new_node
    

    # Allow left-hand-side matrix multiply.
    __rmatmul__ = __matmul__


    def __truediv__(self, other):
        """Dividing two nodes return a new node."""
        if isinstance(other, Node):
            new_node = div_op(self, other)
        else:
            # Divide by a constant stores the constant in the new node's const_attr field.
            new_node = div_byconst_op(self, other)
        return new_node
    

    # Allow left-hand-side division.
    __rtruediv__ = __truediv__

    
    def __pow__(self, other):
        """Exponentiating two nodes return a new node."""
        if isinstance(other, Node):
            new_node = pow_op(self, other)
        else:
            # Exponentiate by a constant stores the constant in the new node's const_attr field.
            new_node = pow_byconst_op(self, other)
        return new_node

    
    # Allow left-hand-side exponentiation.
    __rpow__ = __pow__


    def __str__(self):
        """Allow print to display node name.""" 
        return f"{self.name} = {self.value}"

    __repr__ = __str__


# Op represents operations performed on nodes.
class Op(object):

    def __call__(self):
        # Create a new node and associate the op object with the node.
        new_node = Node()
        new_node.op = self
        return new_node


    def compute(self, node, input_vals):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.

        Returns
        -------
        An output value of the node.
        """
        raise NotImplementedError


    def gradient(self, node, output_grad):
        """Given value of output gradient, compute gradient contributions to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: value of output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError


# Op to element-wise add two nodes.
class AddOp(Op):

    def __call__(self, node_A, node_B):
        # Create a new node that is the result of adding two input nodes.
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value + node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Given gradient of add node, return gradient contributions to each input.
        return [output_grad, output_grad]
    

# Op to element-wise add a nodes by a constant.
class AddByConstOp(Op):

    def __call__(self, node_A, const_val):
        # Create a new node that is the result of adding a node and a constant.
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        new_node.value = node_A.value + const_val
        return new_node

    def gradient(self, node, output_grad):
        # Given gradient of add node, return gradient contribution to input.
        return [output_grad]
    

# Op to element-wise multiply two nodes.
class MulOp(Op):

    def __call__(self, node_A, node_B):
        # Create a new node that is the result of multiplying two input nodes.
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value * node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Given gradient of multiply node, return gradient contributions to each input.
        return [output_grad * node.inputs[1], output_grad * node.inputs[0]]


# Op to element-wise multiply a nodes by a constant.
class MulByConstOp(Op):

    def __call__(self, node_A, const_val):
        # Create a new node that is the result of multiplying a node and a constant.
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        new_node.value = node_A.value * const_val
        return new_node
        
    def gradient(self, node, output_grad):
        # Given gradient of mul by const node, return gradient contributions to the input node.
        return [node.const_attr * output_grad]     
       

# Op to matrix multiply two nodes.
class MatMulOp(Op):

    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        """
        Create a new node that is the result a matrix multiple of two input nodes.

        Parameters
        ----------
        node_A: lhs of matrix multiply
        node_B: rhs of matrix multiply
        trans_A: whether to transpose node_A
        trans_B: whether to transpose node_B
        """
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.name = "MatMul(%s,%s,%s,%s)" % (node_A.name, node_B.name, str(trans_A), str(trans_B))

        # Compute the matrix multiplication based on the transposition attributes
        if trans_A:
            A = node_A.value.T
        else:
            A = node_A.value

        if trans_B:
            B = node_B.value.T
        else:
            B = node_B.value

        new_node.value = np.matmul(A, B)
        return new_node
    

    def gradient(self, node, output_grad):
        """
        Given gradient of multiply node, return gradient contributions to each input.
            
        Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        """
        if node.matmul_attr_trans_A:
            dA = MatMulOp(output_grad, node.inputs[1], trans_A=False, trans_B=True)
        else:
            dA = MatMulOp(output_grad, node.inputs[1], trans_A=False, trans_B=False)

        if node.matmul_attr_trans_B:
            dB = MatMulOp(node.inputs[0], output_grad, trans_A=True, trans_B=False)
        else:
            dB = MatMulOp(node.inputs[0], output_grad, trans_A=False, trans_B=False)

        return [dA, dB]


# Op
class MatMulByConstOp(Op):

    def __call__(self, node_A, const_matrix):
        new_node = Op.__call__(self)
        new_node.const_attr = const_matrix
        new_node.inputs = [node_A]
        new_node.name = "(%s*%s)" % (node_A.name, str(const_matrix))
        new_node.value = np.matmul(node_A.value, const_matrix)
        return new_node

    def gradient(self, node, output_grad):
        # Given gradient of matmul by const node, return gradient contributions to the input node.
        const_matrix = node.const_attr
        grad_A = np.matmul(output_grad, const_matrix.T)
        return [grad_A]


# Op to element-wise divide two nodes.
class DivOp(Op):

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s/%s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value / node_B.value
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of divide node, return gradient contributions to each input."""
        return [output_grad / node.inputs[1], -output_grad * node.inputs[0] / (node.inputs[1] ** 2)]


# Op to element-wise divide a nodes by a constant.
class DivByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s/%s)" % (node_A.name, str(const_val))
        new_node.value = node_A.value / const_val
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of divide by const node, return gradient contributions to the input node."""
        return [output_grad / node.const_attr]
        

# Op to perform element-wise power (exponentiation) of two nodes.
class PowOp(Op):

    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s^%s)" % (node_A.name, node_B.name)
        new_node.value = np.power(node_A.value, node_B.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of pow node, return gradient contributions to each input."""
        A, B = node.inputs[0].value, node.inputs[1].value
        grad_A = output_grad * B * np.power(A, B - 1)
        grad_B = output_grad * np.power(A, B) * np.log(A)
        return [grad_A, grad_B]
    

# Op to perform element-wise power (exponentiation) of a node and a constant.
class PowByConstOp(Op):

    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s^%s)" % (node_A.name, str(const_val))
        new_node.value = node_A.value ** const_val
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of pow by const node, return gradient contributions to the input node."""
        A = node.inputs[0].value
        const_val = node.const_attr
        grad_A = output_grad * const_val * A ** (const_val - 1)
        return [grad_A]



class PlaceholderOp(Op):
    """Op to feed value to a nodes."""

    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)        
        return new_node

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None
    

def Variable(name, value):
    # User defined variables in an expression.  
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    placeholder_node.value = value
    return placeholder_node



class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""

    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        new_node.value = np.zeros(node_A.value.shape)
        return new_node

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


class OnesLikeOp(Op):
    """Op that represents a constant np.ones_like."""

    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        new_node.value = np.ones(node_A.value.shape)
        return new_node


    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]
    

# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
matmul_byconst_op = MatMulByConstOp()
div_op = DivOp()
div_byconst_op = DivByConstOp()
pow_op = PowOp()
pow_byconst_op = PowByConstOp()

placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()


# def gradients(loss_node, nodes):
#     """Compute gradients of nodes with respect to the loss node using backpropagation.

#     Parameters
#     ----------
#     loss_node: Node
#         The output node (scalar) representing the loss.
#     nodes: List[Node]
#         List of input nodes with respect to which the gradients are computed.

#     Returns
#     -------
#     gradients: Dict[Node, float]
#         Dictionary mapping input nodes to their corresponding gradients.
#     """
#     # Initialize gradients dictionary with the loss_node's gradient
#     gradients = {loss_node: 1.0}

#     # Find the topological sort of nodes ending in the loss node
#     topo_order = find_topo_sort([loss_node])

#     # Perform reverse-mode automatic differentiation (backpropagation)
#     for node in reversed(topo_order):
#         # Get the gradient of the current node with respect to its output
#         node_grad = gradients[node]

#         # Get the gradients of the node with respect to its inputs
#         input_gradients = node.op.gradient(node, node_grad)

#         # Update gradients for input nodes of the current node
#         for i, input_node in enumerate(node.inputs):
#             gradients[input_node] = gradients.get(input_node, 0.0) + input_gradients[i]

#     # Collect gradients for the specified input
#     input_gradients = {node: gradients.get(node, 0.0) for node in nodes}

#     return input_gradients

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
    gradients: Dict[Node, float]
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


def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.
    
    A simple algorithm is to do a post-order DFS traversal on the given nodes, 
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)