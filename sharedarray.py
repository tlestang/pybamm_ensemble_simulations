import SharedArray
from concurrent.futures import ProcessPoolExecutor

from sim import Sim


def shm_step(step):
    shm_y = SharedArray.attach("shm://y")
    shm_t = SharedArray.attach("shm://t")
    shm_i_app = SharedArray.attach("shm://i_app")
    ind = step.ind
    inputs = {"Current": shm_i_app[ind]}
    step_solution = step.do_step(shm_y[:, ind], shm_t[ind], inputs)
    shm_y[:, ind] = step_solution.y[:, -1]
    shm_t[ind] = step_solution.t[-1]


def solve_w_SharedArray(model, sol_init, end_time, Nsteps, dt=1, Nspm=8):
    try:
        shm_y = SharedArray.create("shm://y", [sol_init.y.shape[0], Nspm], dtype=float)
        shm_t = SharedArray.create("shm://t", [Nspm], dtype=float)
        shm_i_app = SharedArray.create("shm://i_app", [Nspm], dtype=float)
    except:
        SharedArray.delete("shm://y")
        SharedArray.delete("shm://t")
        SharedArray.delete("shm://i_app")
        shm_y = SharedArray.create("shm://y", [sol_init.y.shape[0], Nspm], dtype=float)
        shm_t = SharedArray.create("shm://t", [Nspm], dtype=float)
        shm_i_app = SharedArray.create("shm://i_app", [Nspm], dtype=float)

    i_app = 1.0
    for i in range(Nspm):
        shm_y[:, i] = sol_init.y[:, -1]
        shm_t[i] = 0.0
        shm_i_app[i] = i_app * (1 + (i + 1) / Nspm)

    time = 0
    tstep = 0
    while time < end_time:
        print("Time", time)
        work = [Sim(model, sol_init, dt, ind, tstep) for ind in range(Nspm)]
        ex = ProcessPoolExecutor()
        ex.map(shm_step, work)
        ex.shutdown(wait=True)
        print(shm_t[:10])
        print(shm_y[0, :10])
        time += dt
        tstep += 1

    SharedArray.delete("shm://y")
    SharedArray.delete("shm://t")
    SharedArray.delete("shm://i_app")
