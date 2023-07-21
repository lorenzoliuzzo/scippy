from mathematics import Variable, gradients
from physics import Quantity
from physics import units as U


def test_variables():
    a = Variable("a", 2)
    b = Variable("b", 1)
    c = a ** 2 + b / 6

    print(a)  
    print(b)  
    print(c)  

    # Compute gradients of 'c' with respect to 'a' and 'b'
    grads = gradients(c, [a, b])

    # Print gradients
    print("Gradient of c:")
    for node, grad in grads.items():
        print(f"wrt {node.name} =", grad)


def test_hyper_variables():
    a = Variable("a", Quantity(2, U.m))
    b = Variable("b", Quantity(1, U.m2))
    c = a ** 2 + b / 6

    print(a)  
    print(b)  
    print(c)  

    # Compute gradients of 'c' with respect to 'a' and 'b'
    grads = gradients(c, [a, b])

    # Print gradients
    print("Gradient of c:")
    for node, grad in grads.items():
        print(f"wrt {node.name} =", grad)


# Test the code
if __name__ == "__main__":
    test_variables()
    test_hyper_variables()