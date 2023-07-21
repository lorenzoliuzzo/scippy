# @file: tests/quantity.py 
# @auth: Lorenzo Liuzzo
# @date: 2021-07-21
# @desc: Test the Quantity class of the physics package

import unittest
from physics import Quantity
from physics import units as U
import math
# from mathematics.functions import sin, cos


class TestQuantityOperations(unittest.TestCase):

    def test_addition(self):
        a = Quantity(2, U.m)
        b = Quantity(1, U.m)
        c = a + b
        self.assertEqual(c.value, 3)  # The numerical value should be 3 (2 + 1)
        self.assertEqual(c.unit, U.m)  # The unit should remain as meters

    def test_subtraction(self):
        a = Quantity(2, U.m)
        b = Quantity(1, U.m)
        c = a - b
        self.assertEqual(c.value, 1)  # The numerical value should be 1 (2 - 1)
        self.assertEqual(c.unit, U.m)  # The unit should remain as meters

    def test_multiplication(self):
        a = Quantity(2, U.m)
        b = Quantity(3, U.s)
        c = a * b
        self.assertEqual(c.value, 6)  # The numerical value should be 6 (2 * 3)
        self.assertEqual(c.unit, U.m * U.s)  # The unit should be meters * seconds

    def test_division(self):
        a = Quantity(10, U.m)
        b = Quantity(2, U.s)
        c = a / b
        self.assertEqual(c.value, 5)  # The numerical value should be 5 (10 / 2)
        self.assertEqual(c.unit, U.m / U.s)  # The unit should be meters per second

    def test_power(self):
        a = Quantity(3, U.m)
        b = 2
        c = a ** b
        self.assertEqual(c.value, 9)  # The numerical value should be 9 (3 ^ 2)
        self.assertEqual(c.unit, U.m ** 2)  # The unit should be meters squared

        d = c ** 0.5
        self.assertEqual(d.value, 3)  # The numerical value should be 3 (9 ^ 0.5)
        self.assertEqual(d.unit, U.m)  # The unit should be meters


    def test_sin(self):
        a = Quantity(0, U.rad)
        b = math.sin(a)
        self.assertEqual(b, 0)

    def test_cos(self):
        a = Quantity(0, U.rad)
        b = math.cos(a)
        self.assertEqual(b, 1)


if __name__ == "__main__":
    unittest.main()
