from multiprocessing import Pool

import numpy as np

from sim import SimPool


def worker(work):
    solution = work.sol_init
    for t in range(0, work.end_time, work.dt):
        solution = work.do_step(solution)

    return solution


def solve_w_pool(model, sol_init, Nsteps, dt=1, Nspm=8):
    end_time = Nsteps * dt
    i_app = 1.0
    list_of_inputs = [{"Current": i_app * (1 + (i + 1) / Nspm)} for i in range(Nspm)]
    work = [
        SimPool(model, sol_init, dt, end_time, list_of_inputs[ind])
        for ind in range(Nspm)
    ]
    p = Pool()
    solutions = p.map(worker, work)

    return np.array([sol.y[:, -1] for sol in solutions]).transpose()
