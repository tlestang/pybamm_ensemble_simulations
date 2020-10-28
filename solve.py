from multiprocessing import Pool

import numpy as np

from sim import SimSolve


def worker(work):
    solver = work.model.default_solver
    return solver.solve(
        work.model,
        t_eval=np.arange(0,work.end_time,work.dt),
        inputs=work.inputs,
    )
    


def solve_w_pool_solve(model, Nsteps, dt, Nspm, processes=None):
    end_time = Nsteps * dt
    i_app = 1.0
    list_of_inputs = [{"Current": i_app * (1 + (i + 1) / Nspm)} for i in range(Nspm)]
    work = [SimSolve(model, dt, end_time, list_of_inputs[ind]) for ind in range(Nspm)]
    with Pool(processes) as p:
        solutions = p.map(worker, work)

    yarray = np.array([sol.y[:, -1] for sol in solutions]).transpose()
    tarray = np.array([sol.t[-1] for sol in solutions])

    return yarray, tarray
