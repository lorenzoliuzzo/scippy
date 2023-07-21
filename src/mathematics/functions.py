import math
import numpy as np
from .autodiff import *


def add(x, y):
    if isinstance(x, Node) and isinstance(y, Node):
        return AddOp()(x, y)
    elif isinstance(x, Node):
        return AddByConstOp(y)(x)
    elif isinstance(y, Node):
        return AddByConstOp(x)(y)
    else:
        return x + y
    

def sub(x, y):
    if isinstance(x, Node) and isinstance(y, Node):
        return SubOp()(x, y)
    elif isinstance(x, Node):
        return SubByConstOp(y)(x)
    elif isinstance(y, Node):
        return SubByConstOp(x)(y)
    else:
        return x - y
    

def mul(x, y):
    if isinstance(x, Node) and isinstance(y, Node):
        return MulOp()(x, y)
    elif isinstance(x, Node):
        return MulByConstOp(y)(x)
    elif isinstance(y, Node):
        return MulByConstOp(x)(y)
    else:
        return x * y
    

def div(x, y):
    if isinstance(x, Node) and isinstance(y, Node):
        return DivOp()(x, y)
    elif isinstance(x, Node):
        return DivByConstOp(y)(x)
    elif isinstance(y, Node):
        return DivByConstOp(x)(y)
    else:
        return x / y
    


def pow(x, y):
    if isinstance(x, Node) and isinstance(y, Node):
        return PowOp()(x, y)
    elif isinstance(x, Node):
        return PowByConstOp(y)(x)
    elif isinstance(y, Node):
        return PowByConstOp(x)(y)
    else:
        return x ** y
    

def neg(x):
    if isinstance(x, Node):
        return NegOp()(x)
    else:
        return -x
    

def abs(x):
    if isinstance(x, Node):
        return AbsOp()(x)
    else:
        return math.abs(x)


def exp(x):
    if isinstance(x, Node):
        return ExpOp()(x)
    else:
        return math.exp(x)
    

def log(x):
    if isinstance(x, Node):
        return LogOp()(x)
    else:
        return math.log(x)
    
    
def sin(x):
    if isinstance(x, Node):
        return SinOp()(x)
    else:
        return math.sin(x)
    
def cos(x):
    if isinstance(x, Node):
        return CosOp()(x)
    else:
        return math.cos(x)
    

def tan(x):
    if isinstance(x, Node):
        return TanOp()(x)
    else:
        return math.tan(x)
    

def sinh(x):
    if isinstance(x, Node):
        return SinhOp()(x)
    else:
        return math.sinh(x)
    

def cosh(x):
    if isinstance(x, Node):
        return CoshOp()(x)
    else:
        return math.cosh(x)
    

def tanh(x):
    if isinstance(x, Node):
        return TanhOp()(x)
    else:
        return math.tanh(x)
    

def asin(x):
    if isinstance(x, Node):
        return AsinOp()(x)
    else:
        return np.arcsin(x)
    

def acos(x):
    if isinstance(x, Node):
        return AcosOp()(x)
    else:
        return np.arccos(x)
    

def atan(x):
    if isinstance(x, Node):
        return AtanOp()(x)
    else:
        return np.arctan(x)
    

def asinh(x):
    if isinstance(x, Node):
        return AsinhOp()(x)
    else:
        return np.arcsinh(x)
    

def acosh(x):
    if isinstance(x, Node):
        return AcoshOp()(x)
    else:
        return np.arccosh(x)
    

def atanh(x):
    if isinstance(x, Node):
        return AtanhOp()(x)
    else:
        return np.arctanh(x)
    
