"""Tests for models."""
from internal import models
from internal import datasets
from internal import configs
from internal import solver
from absl.testing import absltest

import numpy as np

class SolversTest(absltest.TestCase):
  def test_sudoku_solving(self):
    np.random.seed(0)
    config = configs.Config(
        data_file = "dataset/Latin_Square_Problem_HARD.txt",
        data_ext = "txt",
        puzzle_type = "sudoku",
        solver_type = "z3"
    )
    data = datasets.Dataset(config)
    constraint_params = data.constraint_params
    model, solving_constraint = \
        models.construct_model(
          config = config, 
          constraint_param = constraint_params
        )

    solver_ = solver.get_solver_cls(config.solver_type)(
      config = config
    )

    solver_(
      model, 
      solving_constraint=solving_constraint,
      solving_params=constraint_params
    )

  def test_cnf_solving(self):
    np.random.seed(0)
    config = configs.Config(
        data_file = "dataset/test.dimacs",
        data_ext = "dimacs",
        puzzle_type = "cnf",
        solver_type = "z3"
    )
    data = datasets.Dataset(config)
    constraint_params = data.constraint_params
    model, solving_constraint = \
        models.construct_model(
          config = config, 
          constraint_param = constraint_params
        )

    solver_ = solver.get_solver_cls(config.solver_type)(
      config = config
    )
    solver_(
      model, 
      solving_constraint=solving_constraint,
      solving_params=constraint_params
    )
    