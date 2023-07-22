from physics import Quantity, MaterialPoint
from physics import units as U
from physics.potentials import Elastic
from physics import Lagrangian, Hamiltonian
from mathematics import gradients

import numpy as np

mp = MaterialPoint('mp', Quantity(np.array([5, 5, 5]), U.m), Quantity(np.array([-2, -2, -2]), U.m / U.s), Quantity(1, U.kg))
print('\n')
print(mp)
print('\n')

kinetic_energy = mp.mass * np.dot(mp.velocity, mp.velocity) / 2
grad = gradients(kinetic_energy, [mp.velocity, mp.momentum])
print(grad[mp.velocity])
print(grad[mp.momentum])

mp2 = MaterialPoint('mp2', Quantity(np.array([5, 5, 5]), U.m), Quantity(np.array([-2, -2, -2]), U.kg * U.m / U.s), Quantity(1, U.kg))
print('\n')
print(mp2)
print('\n')

kinetic_energy = np.dot(mp2.momentum, mp2.momentum) / (2 * mp2.mass)
grad = gradients(kinetic_energy, [mp2.velocity, mp2.momentum])
print(grad[mp2.velocity])
print(grad[mp2.momentum])

spring = Elastic(Quantity(30, U.N / U.m), Quantity(1, U.m))
print(spring(mp.position))

grad = gradients(spring(mp.position), [mp.position])
print(grad[mp.position])

mp3 = MaterialPoint('mp3', Quantity(np.array([5, 5, 5]), U.m), Quantity(np.array([-2, -2, -2]), U.m / U.s), Quantity(1, U.kg), [spring])
print('\n')
print(mp3)
print('\n')

lagr = Lagrangian(mp3)
lagr_var = lagr()
grad = gradients(lagr_var, [mp3.position, mp3.velocity])
print("Lagrangian:", lagr_var)
print('dL dx: ', grad[mp3.position])
print('dL dv: ', grad[mp3.velocity])

ham = Hamiltonian()
ham.add_body(mp3)
ham_var = ham()
grad = gradients(ham_var, [mp3.position, mp3.momentum])
print("Hamiltonian:", ham_var)
print('dH dx: ', grad[mp3.position])
print('dH dp: ', grad[mp3.momentum])


hamtot = Hamiltonian()
hamtot.add_body(mp)
hamtot.add_body(mp2)
hamtot.add_body(mp3)
hamtot_var = hamtot()
print("\nHamiltonian:", hamtot_var)