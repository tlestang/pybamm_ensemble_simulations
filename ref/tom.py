from concurrent.futures import ProcessPoolExecutor
import numpy as np
import matplotlib.pyplot as plt
import pybamm
import SharedArray
from subprocess import run
import time as time_module

plt.close('all')
pybamm.set_logging_level("WARNING")

def current_function(t):
   return pybamm.InputParameter("Current")

if __name__ == '__main__':
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
   i_app = 1.0

   Nspm = 8
   Nsteps = 10
   try:
       shm_y = SharedArray.create('shm://y', [sol_init.y.shape[0], Nspm], dtype=float)
       shm_t = SharedArray.create('shm://t', [Nspm], dtype=float)
       shm_i_app = SharedArray.create('shm://i_app', [Nspm], dtype=float)
       shm_V = SharedArray.create('shm://V', [Nsteps, Nspm], dtype=float)
   except:
       SharedArray.delete('shm://y')
       SharedArray.delete('shm://t')
       SharedArray.delete('shm://i_app')
       SharedArray.delete('shm://V')
       shm_y = SharedArray.create('shm://y', [sol_init.y.shape[0], Nspm], dtype=float)
       shm_t = SharedArray.create('shm://t', [Nspm], dtype=float)
       shm_i_app = SharedArray.create('shm://i_app', [Nspm], dtype=float)
       shm_V = SharedArray.create('shm://V', [Nsteps, Nspm], dtype=float)

   for i in range(Nspm):
       shm_y[:, i] = sol_init.y[:, -1]
       shm_t[i] = 0.0
       shm_i_app[i] = i_app*(1+(i+1)/Nspm)
   #f = ex.submit(add_one, shm_key)

   # step model
   dt = 1
   time = 0
   end_time = dt*Nsteps
   step_solver = model.default_solver
   step_solution = None


   def shm_step(args):
       ind, tstep = args
       shm_y = SharedArray.attach('shm://y')
       shm_t = SharedArray.attach('shm://t')
       shm_i_app = SharedArray.attach('shm://i_app')
       shm_V = SharedArray.attach('shm://V')
       sol_init.y[:, -1] = shm_y[:, ind]
       sol_init.t[-1] = shm_t[ind]
       inputs = {"Current": shm_i_app[ind]}
       step_solution = step_solver.step(sol_init, model, dt=dt, npts=2,
                                        inputs=inputs, save=False)
       shm_y[:, ind] = step_solution.y[:, -1]
       shm_t[ind] = step_solution.t[-1]
#        shm_V[tstep, ind] = step_solution['Terminal voltage [V]'](step_solution.t[-1])

   step = 0
   while time < end_time:
       print('Time', time)
       st = time_module.time()
       ex = ProcessPoolExecutor()
       ex.map(shm_step, zip(np.arange(Nspm, dtype=int),
                            np.ones(Nspm, dtype=int)*step))
       ex.shutdown(wait=True)
       print(shm_t[:10])
       print(shm_y[0, :10])
       time += dt
       step += 1
       print('Stepping time', np.around(time_module.time()-st, 2), 's')


   # Get current HEAD commint hash
   cmd = ["git", "rev-parse", "HEAD"]
   hash = run(cmd, capture_output=True, encoding="utf-8").stdout
   header = f"Generated with {__file__}, revision {hash}"
   np.savetxt("base_solution.txt", shm_y, header=header)

   SharedArray.delete('shm://y')
   SharedArray.delete('shm://t')
   SharedArray.delete('shm://i_app')
   SharedArray.delete('shm://V')
