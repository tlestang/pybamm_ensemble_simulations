# PyBaMM/regtest/regtest.py
import os
import sys
import time

from numpy import linspace

sys.path.append("../")
from bench import init_model, get_initial_solution
from sharedarray import solve_w_SharedArray

model = init_model()
sol_init = get_initial_solution(model, linspace(0, 1, 2), {"Current": 0.67})
Nspm = 8
Nsteps = 10
dt = 1

st = time.time()
y, t = solve_w_SharedArray(model, sol_init, Nsteps, dt, Nspm, processes=4)
el_time = time.time() - st

print(f"{el_time:.3f}")
