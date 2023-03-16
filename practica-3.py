import os
import numpy as np
import math
from functools import partial
from itertools import accumulate, repeat
from typing import Callable
from scipy.integrate import solve_ivp

def get_vector_field(func: Callable,
                     t0: float,
                     min_x: float,
                     max_x: float,
                     min_y: float,
                     max_y: float,
                     granularity: int,
                     separation: float = 0.8):
    
    vector_field_grid_X = np.linspace(min_x,
                                      max_x,
                                      granularity)

    vector_field_grid_Y = np.linspace(min_y,
                                      max_y,
                                      granularity)

    vector_lenght_X = separation * (max_x - min_x) / granularity

    vector_lenght_Y = separation * (max_y - min_y) / granularity

    vector_field = np.array([(p, q,
                              (vector_lenght_X * fpq[0]
                               / math.sqrt( fpq[0] ** 2 + fpq[1] ** 2)),
                              (vector_lenght_Y * fpq[1]
                               / math.sqrt( fpq[0] ** 2 + fpq[1] ** 2)))
                             for p, q, fpq in [(xi, yi, func(t0, (xi, yi)))
                                       for xi in vector_field_grid_X
                                       for yi in vector_field_grid_Y]])
    return vector_field
        
INITIAL_PARAMETER_MIN_X = -1
INITIAL_PARAMETER_MAX_X = 1
INITIAL_PARAMETER_MIN_Y = -1
INITIAL_PARAMETER_MAX_Y = 1
VECTOR_GRID_GRANULARITY = 51

INITIAL_VALUES = ([[p0, 0.1]
                   for p0 in np.linspace(-1,
                                         1,
                                         19)]
                  +
                  [[0.1, q0]
                   for q0 in np.linspace(-1,
                                         1,
                                         19)])

A_ORBIT_PARAMETER = 3
B_ORBIT_PARAMETER = .5

t0 = 0
tN = 10
delta = 1e-3
t = np.arange(t0, tN, delta)


def general_orbit(t, z, a, b):
    q, p = z
    return [2* p, - (4 * q / a) * (q * q - b)]

orbit = partial(general_orbit,
                a = A_ORBIT_PARAMETER,
                b = B_ORBIT_PARAMETER)

vector_field = get_vector_field(orbit,
                                0,
                                -1,#INITIAL_PARAMETER_MIN_X,
                                1,#INITIAL_PARAMETER_MAX_X,
                                -1,#INITIAL_PARAMETER_MIN_Y,
                                1,#INITIAL_PARAMETER_MAX_Y,
                                VECTOR_GRID_GRANULARITY)

sol = np.column_stack([solve_ivp(orbit, [t0, tN], p0q0,
                                 t_eval=t,
                                 ).y.T
                       for p0q0 in INITIAL_VALUES])

orbits_data = np.column_stack((t, sol))

np.savetxt("vector_field.dat", vector_field,
           header= "# grid_x grix_y dx/dt dy/dt",
           fmt=['%.5f', '%.5f', '%.5f', '%.5f'])

np.savetxt("orbits.dat", orbits_data, fmt="%.5f")
