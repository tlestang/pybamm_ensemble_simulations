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
