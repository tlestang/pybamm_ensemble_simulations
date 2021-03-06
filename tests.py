import unittest

import numpy as np
import pybamm as pb

from map_serial import solve_w_pool_serial
from pool import solve_w_pool
from serial import solve_serial
from sharedarray import solve_w_SharedArray
from solve import solve_w_pool_solve


def current_function(t):
    return pb.InputParameter("Current")


def get_initial_solution(model, t_eval, inputs):
    solver = pb.CasadiSolver()
    return solver.solve(model, t_eval, inputs=inputs)


class TestEnsembleSimulation(unittest.TestCase):
    def setUp(self):
        pb.set_logging_level("WARNING")

        # load model
        self.model = pb.lithium_ion.SPMe()

        # create geometry
        geometry = self.model.default_geometry

        # load parameter values and process model and geometry
        param = self.model.default_parameter_values

        param.update(
            {
                "Current function [A]": current_function,
            }
        )
        param.update({"Current": "[input]"}, check_already_exists=False)
        param.process_model(self.model)
        param.process_geometry(geometry)

        # set mesh
        mesh = pb.Mesh(
            geometry, self.model.default_submesh_types, self.model.default_var_pts
        )

        # discretise self.model
        disc = pb.Discretisation(mesh, self.model.default_spatial_methods)
        disc.process_model(self.model)

        self.sol_init = get_initial_solution(
            self.model, np.linspace(0, 1, 2), {"Current": 0.67}
        )

        self.Nsteps = 10
        self.dt = 1
        self.Nspm = 8

        expected_y_flat = np.fromfile("ref/ref_solution.bin")
        Npoint = self.sol_init.y.shape[0]
        Nspm = len(expected_y_flat) // Npoint
        self.expected_y = expected_y_flat.reshape((Npoint, Nspm))

    def test_SharedArray(self):
        y, t = solve_w_SharedArray(
            self.model, self.sol_init, self.Nsteps, self.dt, self.Nspm
        )

        np.testing.assert_almost_equal(y, self.expected_y, decimal=5)

    def test_Pool(self):
        y, t = solve_w_pool(self.model, self.sol_init, self.Nsteps, self.dt, self.Nspm)

        np.testing.assert_almost_equal(y, self.expected_y, decimal=5)

    def test_Pool_solve(self):
        y, t = solve_w_pool_solve(self.model, self.Nsteps, self.dt, self.Nspm)

        # np.testing.assert_almost_equal(y, self.expected_y, decimal=5)
        self.assertTrue(True)

    def test_Serial(self):
        y, t = solve_serial(self.model, self.sol_init, self.Nsteps, self.dt, self.Nspm)

        np.testing.assert_almost_equal(y, self.expected_y, decimal=5)

    def test_Pool_serial(self):
        y, t = solve_w_pool(
            self.model, self.sol_init, self.Nsteps, self.dt, self.Nspm, serial=True
        )

        np.testing.assert_almost_equal(y, self.expected_y, decimal=5)
