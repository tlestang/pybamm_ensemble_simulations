import pybamm
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

from worker import worker

plt.close('all')
pybamm.set_logging_level("WARNING")

def current_function(t):
   return pybamm.InputParameter("Current")

# load model
model = pybamm.lithium_ion.SPMe()

# create geometry
geometry = model.default_geometry

# load parameter values and process model and geometry
param = model.default_parameter_values

param.update({"Current function [A]": current_function,})
param.update({"Current": "[input]"}, check_already_exists=False)
param.process_model(model)
param.process_geometry(geometry)

# set mesh
mesh = pybamm.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

# discretise model
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)

# solve model
inputs = {"Current": 0.67}
t_eval = np.linspace(0, 1, 2)
solver = pybamm.CasadiSolver()
sol_init = solver.solve(model, t_eval, inputs=inputs)

Nsteps = 10
dt = 1
end_time = dt*Nsteps
Nspm = 8
i_app = 1.0

start_time = time.time()
inputs_list = [{"Current": i_app*(1+(i+1)/Nspm)} for i in range(Nspm)]
p = Pool()
solutions = p.map(worker, zip([model]*Nspm,
                        [sol_init]*Nspm,
                        inputs_list,
                        [end_time]*Nspm))

time = time.time()-start_time
print(f"Overall time: {time}")

vectors = np.array([sol.y[:,-1] for sol in solutions]).transpose()

expected_vectors = np.fromfile("ref/base_solution.bin").reshape(vectors.shape)
np.testing.assert_almost_equal(vectors, expected_vectors, decimal=5)

print("DONE")


   
