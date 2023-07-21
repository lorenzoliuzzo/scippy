# @file: tests/quantity.py 
# @auth: Lorenzo Liuzzo
# @date: 2021-07-21
# @desc: Test the Variable class for automatic differentiation on the Quantity class of the physics package

import unittest
from mathematics import Variable, gradients
from mathematics import functions as fn
from physics import Quantity
from physics import units as U


class TestVariableOperations(unittest.TestCase):

    def test_addition(self):
        a = Variable('a', Quantity(2, U.m))
        b = Variable('b', Quantity(1, U.mm))
        c = a + b
        self.assertEqual(c.name, '(a+b)')
        self.assertEqual(c.value, Quantity(2.001, U.m))  
        

    def test_subtraction(self):
        a = Variable('a', Quantity(2, U.m))
        b = Variable('b', Quantity(1, U.mm))
        c = a - b
        self.assertEqual(c.name, '(a-b)') 
        self.assertEqual(c.value, Quantity(1.999, U.m))


    def test_multiplication(self):
        a = Variable('a', Quantity(2, U.m))
        b = Variable('b', Quantity(3, U.s))
        c = a * b
        self.assertEqual(c.name, '(a*b)')
        self.assertEqual(c.value, Quantity(6, U.m * U.s))


    def test_division(self):
        a = Variable('a', Quantity(2, U.m))
        b = Variable('b', Quantity(3, U.s))
        c = a / b
        self.assertEqual(c.name, '(a/b)')
        self.assertEqual(c.value, Quantity(2 / 3, U.m / U.s))


    def test_power(self):
        a = Quantity(3, U.m)
        b = 2
        c = a ** b
        self.assertEqual(c.value, Quantity(9, U.m2))

        d = c ** 0.5
        self.assertEqual(d.value, Quantity(3, U.m))


    def test_sin(self):
        a = Variable('a', Quantity(0, U.rad))
        b = fn.sin(a)
        self.assertEqual(b.value, 0)


    def test_gradient_sum(self):
        a = Variable("a", Quantity(2, U.m))
        b = Variable("b", Quantity(1, U.m))
        c = a + b

        # Compute gradients of 'c' with respect to 'a' and 'b'
        grads = gradients(c, [a, b])

        self.assertEqual(grads[a], 1)  # The gradient of 'c' with respect to 'a' should be 1
        self.assertEqual(grads[b], 1)  # The gradient of 'c' with respect to 'b' should be 1
        

    def test_gradient_product(self):
        a = Variable("a", Quantity(2, U.m))
        b = Variable("b", Quantity(5, U.m))
        c = a * b

        # Compute gradients of 'c' with respect to 'a' and 'b'
        grads = gradients(c, [a, b])

        self.assertEqual(grads[a].value, b.value)
        self.assertEqual(grads[b].value, a.value)


if __name__ == "__main__":
    unittest.main()
