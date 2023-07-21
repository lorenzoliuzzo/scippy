import unittest
from mathematics import Variable, gradients
from mathematics.functions import exp
from physics import Quantity
from physics import units as U


class TestFunctions(unittest.TestCase):

    def test_exp(self):
        # Test input value
        x = Variable("x", Quantity(0, U.rad))
        y = exp(x)
        g = gradients(y, [x])
        self.assertEqual(y.value, 1)
        self.assertEqual(g[x], 1)


if __name__ == "__main__":
    unittest.main()
