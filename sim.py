class Sim:
    def __init__(self, model, sol_init, dt, ind, tstep):
        self.model = model
        self.sol_init = sol_init
        self.dt = dt
        self.ind = ind
        self.tstep = tstep

        self.step_solver = model.default_solver

    def do_step(self, y, t, inputs):
        self.sol_init.y[:, -1] = y
        self.sol_init.t[-1] = t

        return self.step_solver.step(
            self.sol_init,
            self.model,
            dt=self.dt,
            npts=2,
            inputs=inputs,
            save=False,
        )


class SimPool:
    def __init__(self, model, sol_init, dt, end_time, inputs):
        self.model = model
        self.sol_init = sol_init
        self.dt = dt
        self.end_time = end_time
        self.inputs = inputs

        self.step_solver = model.default_solver

    def do_step(self, sol_init):
        return self.step_solver.step(
            sol_init,
            self.model,
            dt=self.dt,
            npts=2,
            inputs=self.inputs,
            save=False,
        )


class SimSolve:
    def __init__(self, model, dt, end_time, inputs):
        self.model = model
        self.dt = dt
        self.end_time = end_time
        self.inputs = inputs
