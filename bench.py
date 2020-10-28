import argparse
import time

import numpy as np
import pybamm as pb

from map_serial import solve_w_pool_serial
from pool import solve_w_pool
from serial import solve_serial
from sharedarray import solve_w_SharedArray
from solve import solve_w_pool_solve
from table import make_table


def current_function(t):
    return pb.InputParameter("Current")


def get_initial_solution(model, t_eval, inputs):
    solver = pb.CasadiSolver()
    return solver.solve(model, t_eval, inputs=inputs)


def init_model():
    # load model
    model = pb.lithium_ion.SPMe()

    # create geometry
    geometry = model.default_geometry

    # load parameter values and process model and geometry
    param = model.default_parameter_values

    param.update(
        {
            "Current function [A]": current_function,
        }
    )
    param.update({"Current": "[input]"}, check_already_exists=False)
    param.process_model(model)
    param.process_geometry(geometry)

    # set mesh
    mesh = pb.Mesh(geometry, model.default_submesh_types, model.default_var_pts)

    # discretise model
    disc = pb.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)

    return model


def execute_n_times(func, args, n=10, **kwargs):
    elapsed_time = []
    for rep in range(n):
        print(
            f"Executing funtion {func.__name__}, rep {rep+1} of {n}\n"
            f'- with nproc = {kwargs.get("processes")}'
        )
        st = time.time()
        y, t = func(*args, **kwargs)
        elapsed_time.append(time.time() - st)

    return elapsed_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run some benchmarks")
    parser.add_argument("--sharedarray", action="store_true")
    parser.add_argument("--pool", action="store_true")
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--map-serial", action="store_true")
    parser.add_argument("--solve", action="store_true")
    args = parser.parse_args()
    run_all = not (
        args.sharedarray or args.pool or args.serial or args.map_serial or args.solve
    )

    model = init_model()
    sol_init = get_initial_solution(model, np.linspace(0, 1, 2), {"Current": 0.67})
    Nreps = 10
    nproc_range = range(2, 6, 2)
    Nspm = 8
    Nsteps = 10
    dt = 1
    description = (
        "Time the resolution of an ensemble of SPMe models, for various number of processes and report timings. "
        "Each implementation is executed {} times. "
        "This is done following different implementations, see the list of options below. "
        "Nspm = {}, Nsteps = {}, dt = {}, nproc_range = {}".format(
            Nreps, Nspm, Nsteps, dt, list(nproc_range)
        )
    )

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--sharedarray",
        action="store_true",
        help="Use implementation based on SharedArray package.",
    )
    parser.add_argument(
        "--pool",
        action="store_true",
        help="Use implementation based on multiprocessing.Pool.map",
    )
    parser.add_argument(
        "--solve",
        action="store_true",
        help="Same as --pool but worker function calls BaseSolver.solve() instead of stepping the model with BaseSolver.step().",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Serial implementation, i.e. solve SPMs in a sequence.",
    )
    args = parser.parse_args()
    run_all = not (args.sharedarray or args.pool or args.serial or args.solve)

    model = init_model()
    sol_init = get_initial_solution(model, np.linspace(0, 1, 2), {"Current": 0.67})

    solver_args = (model, sol_init, Nsteps, dt, Nspm)
    summary_table_content = {}

    if args.sharedarray or run_all:
        elapsed_time_sharedarray = []
        for nproc in nproc_range:
            elapsed_time_sharedarray.append(
                execute_n_times(
                    solve_w_SharedArray, solver_args, n=Nreps, processes=nproc
                )
            )
        summary_table_content.update({"sharedarray": elapsed_time_sharedarray})

    if args.pool or run_all:
        elapsed_time_pool = []
        for nproc in nproc_range:
            elapsed_time_pool.append(
                execute_n_times(solve_w_pool, solver_args, n=Nreps, processes=nproc)
            )
        summary_table_content.update({"pool": elapsed_time_pool})

    if args.solve or run_all:
        solver_args_solve = (model, Nsteps, dt, Nspm)
        elapsed_time_solve = []
        for nproc in nproc_range:
            elapsed_time_solve.append(
                execute_n_times(
                    solve_w_pool_solve, solver_args_solve, n=Nreps, processes=nproc
                )
            )
        summary_table_content.update({"solve": elapsed_time_solve})

    if args.serial or run_all:
        elapsed_time = execute_n_times(solve_serial, solver_args, n=Nreps)
        summary_table_content.update({"serial": elapsed_time})

    if args.map_serial or run_all:
        elapsed_time_map_serial = execute_n_times(
            solve_w_pool_serial, solver_args, n=Nreps
        )
        summary_table_content.update({"map-serial": elapsed_time_map_serial})

    table = make_table(summary_table_content, nproc_range)

    if args.serial or run_all:
        with open("scaling_serial.txt", "w") as f:
            f.write(" ".join((f"{numvar:.3f}" for numvar in elapsed_time)))

    if args.map_serial or run_all:
        with open("scaling_map_serial.txt", "w") as f:
            f.write(" ".join((f"{numvar:.3f}" for numvar in elapsed_time_map_serial)))

    if args.sharedarray or run_all:
        with open("scaling_sharedarray.txt", "w") as f:
            np.savetxt(f, np.array(elapsed_time_sharedarray), fmt="%.3f", delimiter=",")

    if args.pool or run_all:
        with open("scaling_pool.txt", "w") as f:
            np.savetxt(f, np.array(elapsed_time_pool), fmt="%.3f", delimiter=",")

    if args.solve or run_all:
        with open("scaling_solve.txt", "w") as f:
            np.savetxt(f, np.array(elapsed_time_solve), fmt="%.3f", delimiter=",")

    print(" ")
    print(f"Nreps = {Nreps}", f"Npsm = {Nspm}")
    print(table)
