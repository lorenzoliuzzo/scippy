# file    scipp/physics/measurements/Quantity.py
# @author  Lorenzo Liuzzo (lorenzoliuzzo@outlook.com)
# @brief   This file contains the implementation of the Quantity class.
# @date    2023-07-20
# @copyright Copyright (c) 2023

from .base_quantity import BaseQuantity
from .unit import Unit
import numpy as np
import math


class Quantity:
    # Class representing a physical quantity with a numerical value and a specific unit.


    def __init__(self, value, unit: Unit = Unit(BaseQuantity(0, 0, 0, 0, 0, 0, 0))):
        """
        Construct a Quantity object with a given numerical value and an associated Unit.

        :param value: The numerical value of the quantity.
        :param unit: The Unit object representing the unit of the quantity.
        """
        self.value = value
        self.unit = unit
            

    def __repr__(self):
        # Return the string representation of the Quantity.
        return f"{self.value} {self.unit}"


    def __add__(self, other):
        """
        Add a Quantity object or a numeric value to the current Quantity object.
        Note that the addition with numeric values is possible only if the current Quantity object has a scalar base.

        :param other: The Quantity object or numeric value to be added.
        :return: A new Quantity object representing the result of the addition.
        :raises ValueError: If the operands have incompatible units.
        """
        if isinstance(other, (int, float)):
            if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
                return Quantity(self.value + other, self.unit)
            else:
                raise TypeError(f"Unsupported operands for +: 'Quantity' of base ' {self.unit.base}' and '{type(other).__name__}'")
        
        if isinstance(other, Quantity) and self.unit.base == other.unit.base:
            if self.unit.prefix == other.unit.prefix:
                return Quantity(self.value + other.value, self.unit)
            else:
                return Quantity(self.value + other.value * other.unit.prefix.factor / self.unit.prefix.factor, self.unit)
        else:
            raise TypeError(f"Unsupported operands for +: 'Quantity' of base ' {self.unit.base}' and '{type(other).__name__}' of base ' {other.unit.base}'")


    __radd__ = __add__
    

    def __sub__(self, other):
        """
        Subtract a Quantity object or a numeric value from the current Quantity object.
        Note that the subtraction with numeric values is possible only if the current Quantity object has a scalar base.

        :param other: The Quantity object or numeric value to be subtracted.
        :return: A new Quantity object representing the result of the subtraction.
        :raises ValueError: If the operands have incompatible units.
        """
        if isinstance(other, (int, float)):
            if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
                return Quantity(self.value - other, self.unit)
            else:
                raise TypeError(f"Unsupported operands for -: 'Quantity' of base ' {self.unit.base}' and '{type(other).__name__}'")
        
        if isinstance(other, Quantity) and self.unit.base == other.unit.base:
            if self.unit.prefix == other.unit.prefix:
                return Quantity(self.value - other.value, self.unit)
            else:
                return Quantity(self.value - other.value * other.unit.prefix.factor / self.unit.prefix.factor, self.unit)
        else:
            raise TypeError(f"Unsupported operands for -: 'Quantity' of base ' {self.unit.base}' and '{type(other).__name__}' of base ' {other.unit.base}'")

    def __rsub__(self, other):
        return -self.__sub__(other)
    

    def __mul__(self, other):
        """
        Multiply the current Quantity object with another Quantity object or a numeric value.

        :param other: The Quantity object or numeric value to be multiplied.
        :return: A new Quantity object representing the result of the multiplication.
        :raises ValueError: If the operands have incompatible units.
        """
        if isinstance(other, Quantity):
            return Quantity(self.value * other.value, self.unit * other.unit)
        
        elif isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit)

        raise ValueError("Unsupported operands for *: 'Quantity' and '{}'".format(type(other).__name__))
    
    def __rmul__(self, other):
        return self.__mul__(other)
    

    def __truediv__(self, other):
        """
        Divide the current Quantity object by another Quantity object or a numeric value.

        :param other: The Quantity object or numeric value to divide by.
        :return: A new Quantity object representing the result of the division.
        :raises ValueError: If the operands have incompatible types.
        """
        if isinstance(other, Quantity):
            return Quantity(self.value / other.value, self.unit / other.unit)
        
        elif isinstance(other, (int, float)):
            return Quantity(self.value / other, self.unit)

        raise ValueError("Unsupported operands for /: 'Quantity' and '{}'".format(type(other).__name__))     
    
    def __rtruediv__(self, other):

        if isinstance(other, (int, float)):
            return Quantity(other / self.value, self.unit ** -1)
    
        elif isinstance(other, Quantity):
            return Quantity(other.value / self.value, other.unit / self.unit)

        raise ValueError("Unsupported operands for /: '{}' and 'Quantity'".format(type(other).__name__))
    

    def __pow__(self, other):
        """
        Raise the current Quantity object to the power of another Quantity object or a numeric value.

        :param other: The Quantity object or numeric value representing the exponent.
        :return: A new Quantity object representing the result of the exponentiation.
        :raises ValueError: If the operands have incompatible type.
        """
        if isinstance(other, (int, float)):
            return Quantity(self.value ** other, self.unit ** other)
        
        elif isinstance(other, Quantity):
            return Quantity(self.value ** other.value, self.unit ** other.value)

        raise TypeError("Unsupported operands for **: 'Quantity' and '{}'".format(type(other).__name__))
    

    def to(self, target_unit: Unit):
        """
        Convert the quantity to the specified target unit.

        :param target_unit: The target unit to convert to.
        :return: A new Quantity object with the converted value and unit.
        :raises ValueError: If the conversion is not possible due to incompatible units.
        """
        if self.unit.base != target_unit.base:
            raise ValueError("Incompatible units for conversion.")
        
        # Convert the value to the target unit's prefix
        return Quantity(self.value * self.unit.prefix.factor / target_unit.prefix.factor, target_unit)


    # Additional arithmetic operations
    def __floordiv__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value // other.value, self.unit / other.unit)
        elif isinstance(other, (int, float)):
            return Quantity(self.value // other, self.unit)
        raise ValueError("Unsupported operands for //: 'Quantity' and '{}'".format(type(other).__name__))

    def __mod__(self, other):
        if isinstance(other, Quantity):
            return Quantity(self.value % other.value, self.unit)
        elif isinstance(other, (int, float)):
            return Quantity(self.value % other, self.unit)
        raise ValueError("Unsupported operands for %: 'Quantity' and '{}'".format(type(other).__name__))


    def __rfloordiv__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(other // self.value, self.unit ** -1)
        elif isinstance(other, Quantity):
            return Quantity(other.value // self.value, other.unit / self.unit)
        raise ValueError("Unsupported operands for //: '{}' and 'Quantity'".format(type(other).__name__))
    
    def __rmod__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(other % self.value, self.unit)
        elif isinstance(other, Quantity):
            return Quantity(other.value % self.value, other.unit)
        raise ValueError("Unsupported operands for %: '{}' and 'Quantity'".format(type(other).__name__))
    

    def __abs__(self):
        return Quantity(abs(self.value), self.unit)
    

    def __neg__(self):
        return self * -1    
    

    def __pos__(self):
        return self
    

    def __inv__(self):
        return Quantity(1 / self.value, self.unit ** -1)
    

    def __exp__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.exp(self.value), self.unit)
        raise TypeError("Unsupported operand for exp: 'Quantity' of base '{}'".format(self.unit.base))
    

    def __log__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.log(self.value), self.unit)
        raise TypeError("Unsupported operand for log: 'Quantity' of base '{}'".format(self.unit.base))


    def __sin__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.sin(self.value), self.unit)
        raise TypeError("Unsupported operand for sin: 'Quantity' of base '{}'".format(self.unit.base))
    

    def __cos__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.cos(self.value), self.unit)
        raise TypeError("Unsupported operand for cos: 'Quantity' of base '{}'".format(self.unit.base))


    def __tan__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.tan(self.value), self.unit)
        raise TypeError("Unsupported operand for tan: 'Quantity' of base '{}'".format(self.unit.base))


    def __asin__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.asin(self.value), self.unit)
        raise TypeError("Unsupported operand for asin: 'Quantity' of base '{}'".format(self.unit.base))


    def __acos__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.acos(self.value), self.unit)
        raise TypeError("Unsupported operand for acos: 'Quantity' of base '{}'".format(self.unit.base))


    def __atan__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.atan(self.value), self.unit)
        raise TypeError("Unsupported operand for atan: 'Quantity' of base '{}'".format(self.unit.base))


    def __sinh__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.sinh(self.value), self.unit)
        raise TypeError("Unsupported operand for sinh: 'Quantity' of base '{}'".format(self.unit.base))
    

    def __cosh__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.cosh(self.value), self.unit)
        raise TypeError("Unsupported operand for cosh: 'Quantity' of base '{}'".format(self.unit.base))
    

    def __tanh__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            return Quantity(math.tanh(self.value), self.unit)
        raise TypeError("Unsupported operand for tanh: 'Quantity' of base '{}'".format(self.unit.base))
    

    def __asinh__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            if self.value >= 0:
                return Quantity(math.log(self.value + math.sqrt(self.value ** 2 + 1)), self.unit)
            else:
                return Quantity(math.log(-self.value + math.sqrt(self.value ** 2 + 1)), self.unit)
        raise TypeError("Unsupported operand for asinh: 'Quantity' of base '{}'".format(self.unit.base))

    def __acosh__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            if self.value >= 1:
                return Quantity(math.log(self.value + math.sqrt(self.value ** 2 - 1)), self.unit)
            else:
                raise ValueError("math domain error")
        raise TypeError("Unsupported operand for acosh: 'Quantity' of base '{}'".format(self.unit.base))
            

    def __atanh__(self):
        if self.unit.base == BaseQuantity(0, 0, 0, 0, 0, 0, 0):
            if abs(self.value) < 1:
                return Quantity(0.5 * math.log((1 + self.value) / (1 - self.value)), self.unit)
            else:
                raise ValueError("math domain error")
        raise TypeError("Unsupported operand for atanh: 'Quantity' of base '{}'".format(self.unit.base))
   

    def __round__(self, n=None):
        return Quantity(round(self.value, n), self.unit)
    

    def __floor__(self):
        return Quantity(math.floor(self.value), self.unit)
    

    def __ceil__(self):
        return Quantity(math.ceil(self.value), self.unit)
    

    def __trunc__(self):
        return Quantity(math.trunc(self.value), self.unit)
    

    def __eq__(self, other):
        if isinstance(other, Quantity):
            return self.value == other.value and self.unit == other.unit
        elif isinstance(other, (int, float)):
            return self.value == other
        raise ValueError("Unsupported operands for ==: 'Quantity' and '{}'".format(type(other).__name__))
    

    def __ne__(self, other):
        if isinstance(other, Quantity):
            return self.value != other.value or self.unit != other.unit
        elif isinstance(other, (int, float)):
            return self.value != other
        raise ValueError("Unsupported operands for !=: 'Quantity' and '{}'".format(type(other).__name__))
    

    def __lt__(self, other):
        if isinstance(other, Quantity):
            return self.value < other.value
        elif isinstance(other, (int, float)):
            return self.value < other
        raise ValueError("Unsupported operands for <: 'Quantity' and '{}'".format(type(other).__name__))
    

    def __le__(self, other):
        if isinstance(other, Quantity):
            return self.value <= other.value
        elif isinstance(other, (int, float)):
            return self.value <= other
        raise ValueError("Unsupported operands for <=: 'Quantity' and '{}'".format(type(other).__name__))
    

    def __gt__(self, other):
        if isinstance(other, Quantity):
            return self.value > other.value
        elif isinstance(other, (int, float)):
            return self.value > other
        raise ValueError("Unsupported operands for >: 'Quantity' and '{}'".format(type(other).__name__))
    

    def __ge__(self, other):

        if isinstance(other, Quantity):
            return self.value >= other.value
        elif isinstance(other, (int, float)):
            return self.value >= other
        raise ValueError("Unsupported operands for >=: 'Quantity' and '{}'".format(type(other).__name__))
    

    def __int__(self):
        return int(self.value * self.unit.prefix.factor)
    

    def __float__(self):
        return float(self.value * self.unit.prefix.factor)


    # convert to array
    def __array__(self):
        return np.array(self.value * self.unit.prefix.factor)
