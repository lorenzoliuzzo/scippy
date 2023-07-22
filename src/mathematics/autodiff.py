import numpy as np
import math as math


class NodeDict(dict):
    def __getitem__(self, key):
        if isinstance(key, Node):
            return super().__getitem__(key)
        for k in self:
            if k == key:
                return super().__getitem__(k)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if isinstance(key, Node):
            super().__setitem__(key, value)
        else:
            for k in self:
                if k == key:
                    super().__setitem__(k, value)
                    return
            raise KeyError(key)

    def __contains__(self, key):
        if isinstance(key, Node):
            return super().__contains__(key)
        for k in self:
            if k == key:
                return True
        return False

    def __delitem__(self, key):
        if isinstance(key, Node):
            super().__delitem__(key)
        else:
            for k in list(self):
                if k == key:
                    super().__delitem__(k)
                    return
            raise KeyError(key)


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
    

    def __sub__(self, other):
        """Subtracting two nodes return a new node."""
        if isinstance(other, Node):
            new_node = sub_op(self, other)
        else:
            # Subtract by a constant stores the constant in the new node's const_attr field.
            new_node = sub_byconst_op(self, other)
        return new_node


    def __rsub__(self, other):  
        """Subtracting a node from a constant return a new node."""
        if isinstance(other, Node):
            new_node = sub_op(other, self)
        else:
            # Subtract a constant from a node stores the constant in the new node's const_attr field.
            new_node = add_byconst_op(-self, other)
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


    def __neg__(self):
        return Node(f'-{self.name}', -self.value)
    

    def __pos__(self):
        return self
    

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
    

    def __square__(self):
        """Element-wise square function."""
        return pow_op(self, 2)
    

    # Logical operators
    def __eq__(self, other):
        """Element-wise equality comparison."""
        return equal_op(self, other)

    def __ne__(self, other):
        """Element-wise inequality comparison."""
        return not_equal_op(self, other)

    def __lt__(self, other):
        """Element-wise less-than comparison."""
        return less_op(self, other)

    def __le__(self, other):
        """Element-wise less-than-or-equal-to comparison."""
        return less_equal_op(self, other)

    def __gt__(self, other):
        """Element-wise greater-than comparison."""
        return greater_op(self, other)

    def __ge__(self, other):
        """Element-wise greater-than-or-equal-to comparison."""
        return greater_equal_op(self, other)
    

    def __sqrt__(self):
        """Element-wise square root function."""
        return pow_op(self, 0.5)
    

    def __neg__(self):
        """Element-wise negative function."""
        return neg_op(self)
    
    def __abs__(self):
        """Element-wise absolute function."""
        return abs_op(self)


    def __log__(self):
        """Element-wise natural logarithm function."""
        return log_op(self)
    
    def __exp__(self):
        """Element-wise exponential function."""
        return exp_op(self)
    

    def __sin__(self):
        """Element-wise sine function."""
        return sin_op(self)
    
    def __cos__(self):
        """Element-wise cosine function."""
        return cos_op(self)
    
    def __tan__(self):
        """Element-wise tangent function."""
        return tan_op(self)


    def __sinh__(self):
        """Element-wise hyperbolic sine function."""
        return sinh_op(self)
    
    def __cosh__(self):
        """Element-wise hyperbolic cosine function."""
        return cosh_op(self)
    
    def __tanh__(self):
        """Element-wise hyperbolic tangent function."""
        return tanh_op(self)


    def __asin__(self):
        """Element-wise arcsine function."""
        return asin_op(self)

    def __acos__(self):
        """Element-wise arccosine function."""
        return acos_op(self)

    def __atan__(self):
        """Element-wise arctangent function."""
        return atan_op(self)


    def __asinh__(self):
        """Element-wise inverse hyperbolic sine function."""
        return asinh_op(self)

    def __acosh__(self):
        """Element-wise inverse hyperbolic cosine function."""
        return acosh_op(self)

    def __atanh__(self):
        """Element-wise inverse hyperbolic tangent function."""
        return atanh_op(self)
    
        
    # def __sigmoid__(self):
    #     """Element-wise sigmoid function."""
    #     return sigmoid_op(self)
    
        
    def __str__(self):
        """Allow print to display node name.""" 
        return f"{self.name} = {self.value}"

    __repr__ = __str__


    def __getattr__(self, attr):
        """Intercept calls to undefined attributes (e.g., math functions) and redirect to Op methods."""
        if hasattr(Op, f"__{attr}__"):
            op_method = getattr(Op, f"__{attr}__")
            return op_method(self)
        raise AttributeError(f"'Node' object has no attribute '{attr}'")
    

    def __hash__(self):
        return id(self)
    
    
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc in (np.add, np.subtract, np.multiply, np.true_divide, np.power):
            return getattr(self, method)(*inputs, **kwargs)
        elif ufunc is np.sin:
            return self.__sin__()
        elif ufunc is np.cos:
            return self.__cos__()
        elif ufunc is np.tan:
            return self.__tan__()
        elif ufunc is np.arcsin:
            return self.__asin__()
        elif ufunc is np.arccos:
            return self.__acos__()
        elif ufunc is np.arctan:
            return self.__atan__()
        elif ufunc is np.arcsinh:
            return self.__asinh__()
        elif ufunc is np.arccosh:
            return self.__acosh__()
        elif ufunc is np.arctanh:
            return self.__atanh__()
        elif ufunc is np.sqrt:
            return self.__sqrt__()
        elif ufunc is np.negative:
            return self.__neg__()
        elif ufunc is np.abs:
            return self.__abs__()
        elif ufunc is np.log:
            return self.__log__()
        elif ufunc is np.exp:
            return self.__exp__()
        return NotImplemented
    

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


# Op to feed value to a nodes.
class PlaceholderOp(Op):

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
    

# Op to element-wise sub two nodes.
class SubOp(Op):

    def __call__(self, node_A, node_B):
        # Create a new node that is the result of subtracting two input nodes.
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s-%s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value - node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Given gradient of sub node, return gradient contributions to each input.
        return [-output_grad, -output_grad]
    

# Op to element-wise sub a nodes by a constant.
class SubByConstOp(Op):

    def __call__(self, node_A, const_val):
        # Create a new node that is the result of subtracting a node and a constant.
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.name = "(%s-%s)" % (node_A.name, str(const_val))
        new_node.value = node_A.value - const_val
        return new_node

    def gradient(self, node, output_grad):
        # Given gradient of sub node, return gradient contribution to input.
        return [-output_grad]
    

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


# Op to element-wise logical AND two nodes.
class AndOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s and %s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value and node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Logical AND is not differentiable, so return None for both inputs.
        return None, None


# Op to element-wise logical OR two nodes.
class OrOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s or %s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value or node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Logical OR is not differentiable, so return None for both inputs.
        return None, None


# Op to element-wise logical NOT a node.
class NotOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "not %s" % node_A.name
        new_node.value = not node_A.value
        return new_node

    def gradient(self, node, output_grad):
        # Logical NOT is not differentiable, so return None for the input.
        return None


# Op to element-wise logical greater-than-or-equal-to comparison of two nodes.
class EqOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Node(f"({node_A.name}=={node_B.name})", np.array_equal(node_A.value, node_B.value))
        new_node.inputs = [node_A, node_B]
        return new_node

    def gradient(self, node, output_grad):
        # Logical >= is not differentiable, so return None for both inputs.
        return None, None

# Op to element-wise logical greater-than comparison of two nodes.
class GtOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s > %s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value > node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Logical > is not differentiable, so return None for both inputs.
        return None, None


# Op to element-wise logical less-than comparison of two nodes.
class LtOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s < %s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value < node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Logical < is not differentiable, so return None for both inputs.
        return None, None


# Op to element-wise logical greater-than-or-equal-to comparison of two nodes.
class GeOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s >= %s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value >= node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Logical >= is not differentiable, so return None for both inputs.
        return None, None


# Op to element-wise logical less-than-or-equal-to comparison of two nodes.
class LeOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "(%s <= %s)" % (node_A.name, node_B.name)
        new_node.value = node_A.value <= node_B.value
        return new_node

    def gradient(self, node, output_grad):
        # Logical <= is not differentiable, so return None for both inputs.
        return None, None


# Op for element-wise negative function.
class NegOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the negative of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "-(%s)" % node_A.name
        new_node.value = -node_A.value
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of negative node, return gradient contributions to the input node."""
        return [-output_grad]
    

# Op for element-wise absolute function.
class AbsOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the absolute value of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "abs(%s)" % node_A.name
        new_node.value = math.abs(node_A.value)
        return new_node
    
    def gradient(self, node, output_grad):
        """Given gradient of absolute node, return gradient contributions to the input node."""
        return [output_grad * math.sign(node.inputs[0].value)]
    

# Op for element-wise exponential function.
class ExpOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the exponential of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "exp(%s)" % node_A.name
        new_node.value = math.exp(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of exponential node, return gradient contributions to the input node."""
        return [output_grad * math.exp(node.inputs[0].value)]


# Op for element-wise natural logarithm function.
class LogOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the natural logarithm of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "log(%s)" % node_A.name
        new_node.value = math.log(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of log node, return gradient contributions to the input node."""
        return [output_grad / node.inputs[0].value]


# Op for element-wise sine function.
class SinOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the sine of node_A."""
        if isinstance(node_A, Node):
            new_node = Node(f"sin({node_A.name})", math.sin(node_A.value))
            new_node.inputs = [node_A]
            new_node.op = SinOp()
            return new_node
        else:
            return math.sin(node_A)

    def gradient(self, node, output_grad):
        """Given gradient of sine node, return gradient contributions to the input node."""
        return [output_grad * math.cos(node.inputs[0].value)]


# Op for element-wise cosine function.
class CosOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the cosine of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "cos(%s)" % node_A.name
        new_node.value = math.cos(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of cosine node, return gradient contributions to the input node."""
        return [-output_grad * math.sin(node.inputs[0].value)]


# Op for element-wise tangent function.
class TanOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the tangent of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "tan(%s)" % node_A.name
        new_node.value = math.tan(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of tangent node, return gradient contributions to the input node."""
        return [output_grad / math.cos(node.inputs[0].value) ** 2]


# Op for element-wise hyperbolic sine function.
class SinhOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the hyperbolic sine of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "sinh(%s)" % node_A.name
        new_node.value = math.sinh(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of hyperbolic sine node, return gradient contributions to the input node."""
        return [output_grad * math.cosh(node.inputs[0].value)]


# Op for element-wise hyperbolic cosine function.
class CoshOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the hyperbolic cosine of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "cosh(%s)" % node_A.name
        new_node.value = math.cosh(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of hyperbolic cosine node, return gradient contributions to the input node."""
        return [output_grad * math.sinh(node.inputs[0].value)]


# Op for element-wise hyperbolic tangent function.
class TanhOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the hyperbolic tangent of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "tanh(%s)" % node_A.name
        new_node.value = math.tanh(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of hyperbolic tangent node, return gradient contributions to the input node."""
        return [output_grad * (1 - math.tanh(node.inputs[0].value) ** 2)]


# Op for element-wise arcsine function.
class AsinOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the arcsine of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "asin(%s)" % node_A.name
        new_node.value = np.arcsin(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of arcsine node, return gradient contributions to the input node."""
        return [output_grad / np.sqrt(1 - node.inputs[0].value ** 2)]


# Op for element-wise arccosine function.
class AcosOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the arccosine of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "acos(%s)" % node_A.name
        new_node.value = np.arccos(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of arccosine node, return gradient contributions to the input node."""
        return [-output_grad / np.sqrt(1 - node.inputs[0].value ** 2)]


# Op for element-wise arctangent function.
class AtanOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the arctangent of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "atan(%s)" % node_A.name
        new_node.value = np.arctan(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of arctangent node, return gradient contributions to the input node."""
        return [output_grad / (1 + node.inputs[0].value ** 2)]


# Op for element-wise inverse hyperbolic sine function.
class AsinhOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the inverse hyperbolic sine of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "asinh(%s)" % node_A.name
        new_node.value = np.arcsinh(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of inverse hyperbolic sine node, return gradient contributions to the input node."""
        return [output_grad / np.sqrt(node.inputs[0].value ** 2 + 1)]


# Op for element-wise inverse hyperbolic cosine function.
class AcoshOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the inverse hyperbolic cosine of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "acosh(%s)" % node_A.name
        new_node.value = np.arccosh(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of inverse hyperbolic cosine node, return gradient contributions to the input node."""
        return [output_grad / np.sqrt(node.inputs[0].value ** 2 - 1)]


# Op for element-wise inverse hyperbolic tangent function.
class AtanhOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents the inverse hyperbolic tangent of node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "atanh(%s)" % node_A.name
        new_node.value = np.arctanh(node_A.value)
        return new_node

    def gradient(self, node, output_grad):
        """Given gradient of inverse hyperbolic tangent node, return gradient contributions to the input node."""
        return [output_grad / (1 - node.inputs[0].value ** 2)]


# Op that represents a constant np.zeros_like.
class ZerosLikeOp(Op):

    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        new_node.value = np.zeros(node_A.value.shape)
        return new_node

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]


# Op that represents a constant np.ones_like.
class OnesLikeOp(Op):

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
sub_op = SubOp()
mul_op = MulOp()
matmul_op = MatMulOp()
div_op = DivOp()
pow_op = PowOp()

add_byconst_op = AddByConstOp()
sub_byconst_op = SubByConstOp()
mul_byconst_op = MulByConstOp()
matmul_byconst_op = MatMulByConstOp()
div_byconst_op = DivByConstOp()
pow_byconst_op = PowByConstOp()

equal_op = EqOp()
and_op = AndOp()
or_op = OrOp()
not_op = NotOp()
gt_op = GtOp()
lt_op = LtOp()
ge_op = GeOp()
le_op = LeOp()

neg_op = NegOp()
abs_op = AbsOp()

exp_op = ExpOp()
log_op = LogOp()

sin_op = SinOp()
cos_op = CosOp()
tan_op = TanOp()

sinh_op = SinhOp()
cosh_op = CoshOp()
tanh_op = TanhOp()

asin_op = AsinOp()
acos_op = AcosOp()
atan_op = AtanOp()

asinh_op = AsinhOp()
acosh_op = AcoshOp()
atanh_op = AtanhOp()

placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()