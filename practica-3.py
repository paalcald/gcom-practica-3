import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

q0 = .4
p0 = .185
t0 = 0
tN = 100
delta = 10 ** (-4)
a = 3
b = 1 / 2
t = np.arange(t0, tN, delta)


def orb(t, z, a, b):
    q, p = z
    return [2* p, - (4 / a) * q * (q * q - b)]


sol = solve_ivp(orb, [t0, tN], [q0, p0],
                dense_output = True,
                args=(a, b))

z = sol.sol(t)

plt.plot(t, z.T)
plt.xlabel('t')
plt.legend(['q', 'p'], shadow=True)
plt.title('Oscilator')
plt.savefig('orb1.png')
plt.close()
print(os.getcwd())
