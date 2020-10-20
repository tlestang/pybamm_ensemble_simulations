from pybamm import CasadiSolver

solver  = CasadiSolver()
def worker(args):
    model, sol_init, inputs, end_time = args

    dt = 1
    solution = sol_init
    for t in range(0, end_time, dt):
        solution = solver.step(solution, model, dt=dt, npts=2,
                               inputs=inputs, save=False)
    return solution
