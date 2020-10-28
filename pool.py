from multiprocessing import Pool, current_process
import time

import numpy as np

from sim import SimPool


def worker(work):
    solution = work.sol_init
    for t in range(0, work.end_time, work.dt):
        solution = work.do_step(solution)

    return solution


def worker_feedback(work):
    st = time.time()
    solution = work.sol_init
    for t in range(0, work.end_time, work.dt):
        solution = work.do_step(solution)

    return (current_process().name, (time.time() - st))


def solve_w_pool(model, sol_init, Nsteps, dt, Nspm, processes=None, serial=False, feedback=False):
    end_time = Nsteps * dt
    i_app = 1.0
    list_of_inputs = [{"Current": i_app * (1 + (i + 1) / Nspm)} for i in range(Nspm)]
    work = [
        SimPool(model, sol_init, dt, end_time, list_of_inputs[ind])
        for ind in range(Nspm)
    ]

    func = worker_feedback if feedback else worker

    if serial:
        solutions = list(map(func, work))
    else:
        with Pool(processes) as p:
            solutions = p.map(func, work, chunksize=Nspm/processes)

    if feedback:
        return solutions

    yarray = np.array([sol.y[:, -1] for sol in solutions]).transpose()
    tarray = np.array([sol.t[-1] for sol in solutions])

    return yarray, tarray
