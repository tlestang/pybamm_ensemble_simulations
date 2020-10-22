import time

import numpy as np
import pybamm as pb

from sharedarray import solve_w_SharedArray
from pool import solve_w_pool

# pb.set_logging_level("WARNING")


def current_function(t):
    return pb.InputParameter("Current")


def get_initial_solution(model, t_eval, inputs):
    solver = pb.CasadiSolver()
    return solver.solve(model, t_eval, inputs=inputs)


# load model
model = pb.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values

param.update(
    {
        "Current function [A]": current_function,
    }
)
param.update({"Current": "[input]"}, check_already_exists=False)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pb.Mesh(
    geometry, model.default_submesh_types, model.default_var_pts
)

# discretise model
disc = pb.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

sol_init = get_initial_solution(
    model, np.linspace(0, 1, 2), {"Current": 0.67}
)

Nsteps = 10
dt = 1

st = time.time()
y, t = solve_w_SharedArray(model, sol_init, Nsteps, dt)
elapsed_time = time.time() - st
print(f"SharedArray: {elapsed_time} s")

st = time.time()
y = solve_w_pool(model, sol_init, Nsteps, dt)
elapsed_time = time.time() - st
print(f"Pool: {elapsed_time} s")
