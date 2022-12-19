"""Tests for solvers."""

from internal import models
from internal import datasets
from internal import configs
from internal import solver
from internal import utils
from absl.testing import absltest
import numpy as np


class QCSolversTest(absltest.TestCase):

  def test_latin_square_solving(self):
    np.random.seed(0)
    for test_file in ("dataset/Latin_Square_Problem_EASY.txt",
                      "dataset/Latin_Square_Problem_MEDIUM.txt",
                      "dataset/Latin_Square_Problem_HARD.txt"):
      config = configs.Config(
          data_file=test_file,
          data_ext="txt",
          puzzle_type="latin_square",
          solver_type="qiskit",
          problem_type="sat")
      data = datasets.Dataset(config)
      constraint_params = data.constraint_params
      model, solving_constraint = \
          models.construct_model(
              config = config,
              constraint_param = constraint_params
          )
      rows, cols = constraint_params['num_rows'], constraint_params['num_cols']
      cfg_solver = solver.get_solver_cls(config.solver_type)(config=config)
      s = cfg_solver(
          model,
          solving_constraint=solving_constraint,
          solving_params=constraint_params,
          ret_solution=True)
      ls_sol = utils.puzzle_sol_str_to_list(s, n_rows=rows, n_cols=cols)
      self.assertTrue(utils.latin_square_verify(ls_sol, n_rows=rows))
