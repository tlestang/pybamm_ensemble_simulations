import time

import numpy as np
from prettytable import PrettyTable
import pybamm as pb

from pool import solve_w_pool
from serial import solve_serial
from sharedarray import solve_w_SharedArray


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


def make_table(benchmark_results, nproc_range):
    impl_col = []
    nproc_col = []
    avg_col = []
    err_col = []
    for impl in benchmark_results.keys():
        if impl == "serial":
            impl_col.append(impl)
            nproc_col.append("-")
            avg_col.append(np.mean(benchmark_results[impl]))
            err_col.append(
                np.std(benchmark_results[impl]) / np.sqrt(len(benchmark_results[impl]))
            )
        else:
            impl_col.extend([impl] * len(list(nproc_range)))
            nproc_col.extend([nproc for nproc in nproc_range])
            avg_col.extend(
                [
                    np.mean(individual_times)
                    for individual_times in benchmark_results[impl]
                ]
            )
            err_col.extend(
                [
                    np.std(individual_times) / np.sqrt(len(individual_times))
                    for individual_times in benchmark_results[impl]
                ]
            )
    table = PrettyTable()
    table.add_column("Implementation", impl_col)
    table.add_column("# processes", nproc_col)
    table.add_column("Average time (s)", avg_col)
    table.add_column("Standard error (s)", err_col)

    return table


if __name__ == "__main__":
    model = init_model()
    sol_init = get_initial_solution(model, np.linspace(0, 1, 2), {"Current": 0.67})
    Nreps = 10
    Nspm = 8
    Nsteps = 10
    dt = 1

    args = (model, sol_init, Nsteps, dt, Nspm)

    elapsed_time = execute_n_times(solve_serial, args, n=Nreps)

    elapsed_time_sharedarray = []
    for nproc in nproc_range:
        elapsed_time_sharedarray.append(
            execute_n_times(solve_w_SharedArray, args, n=Nreps, processes=nproc)
        )

    elapsed_time_pool = []
    for nproc in nproc_range:
        elapsed_time_pool.append(
            execute_n_times(solve_w_pool, args, n=Nreps, processes=nproc)
        )

    table = make_table(
        {
            "SharedArray": elapsed_time_sharedarray,
            "multiprocessing.Pool": elapsed_time_pool,
            "serial": elapsed_time,
        },
        nproc_range,
    )

    with open("scaling_serial.txt", "w") as f:
        f.write(" ".join((f"{numvar:.3f}" for numvar in elapsed_time)))

    with open("scaling_sharedarray.txt", "w") as f:
        np.savetxt(f, np.array(elapsed_time_sharedarray), fmt="%.3f", delimiter=",")

    with open("scaling_pool.txt", "w") as f:
        np.savetxt(f, np.array(elapsed_time_pool), fmt="%.3f", delimiter=",")

    print(" ")
    print(f"Nreps = {Nreps}", "Npsm = {Nspm}")
    print(table)
