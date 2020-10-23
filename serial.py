import numpy as np


def solve_serial(model, sol_init, Nsteps, dt, Nspm):
    end_time = Nsteps * dt
    i_app = 1.0
    list_of_inputs = [{"Current": i_app * (1 + (i + 1) / Nspm)} for i in range(Nspm)]

    yarray = np.zeros((sol_init.y.shape[0], Nspm))
    tarray = np.zeros(Nspm)
    for i in range(Nspm):
        solution = sol_init
        for t in range(0, end_time, dt):
            solution = model.default_solver.step(
                solution,
                model,
                dt=dt,
                npts=2,
                inputs=list_of_inputs[i],
                save=False,
            )
        yarray[:, i] = solution.y[:, -1]
        tarray[i] = solution.t[-1]

    return yarray, tarray
