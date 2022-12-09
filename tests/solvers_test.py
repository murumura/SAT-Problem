"""Tests for solvers."""
from internal import models
from internal import datasets
from internal import configs
from internal import solver
from internal import utils
from absl.testing import absltest

import numpy as np


class SolversTest(absltest.TestCase):

  def test_sudoku_solving(self):
    np.random.seed(0)
    config = configs.Config(
        data_file="dataset/sudoku.txt",
        data_ext="txt",
        puzzle_type="sudoku",
        solver_type="z3",
        problem_type="smt")
    data = datasets.Dataset(config)
    constraint_params = data.constraint_params
    model, solving_constraint = \
        models.construct_model(
          config = config,
          constraint_param = constraint_params
        )

    solver_ = solver.get_solver_cls(config.solver_type)(config=config)
    rows, cols = constraint_params['num_rows'], constraint_params['num_cols']
    s = solver_(
        model,
        solving_constraint=solving_constraint,
        solving_params=constraint_params,
        ret_solution=True)
    sudoku_sol = utils.puzzle_sol_str_to_list(s, n_rows=rows, n_cols=cols)

    self.assertTrue(utils.sudoku_verify(sudoku_sol, n_rows=rows, n_cols=cols))

  def test_latin_square_solving(self):
    np.random.seed(0)
    config = configs.Config(
        data_file="dataset/sudoku.txt",
        data_ext="txt",
        puzzle_type="latin_square",
        solver_type="z3",
        problem_type="smt")
    data = datasets.Dataset(config)
    constraint_params = data.constraint_params
    model, solving_constraint = \
        models.construct_model(
          config = config,
          constraint_param = constraint_params
        )
    rows, cols = constraint_params['num_rows'], constraint_params['num_cols']
    solver_ = solver.get_solver_cls(config.solver_type)(config=config)
    s = solver_(
        model,
        solving_constraint=solving_constraint,
        solving_params=constraint_params,
        ret_solution=True)
    ls_sol = utils.puzzle_sol_str_to_list(s, n_rows=rows, n_cols=cols)
    self.assertTrue(utils.latin_square_verify(ls_sol, n_rows=rows))

  def test_cnf_solving(self):
    np.random.seed(0)
    for test_file in ("dataset/test.dimacs", "dataset/test2.dimacs",
                      "dataset/test3.dimacs", "dataset/test4.dimacs"):
      config = configs.Config(
          data_file=test_file,
          data_ext="dimacs",
          puzzle_type="cnf",
          solver_type="z3",
          problem_type="sat")
      data = datasets.Dataset(config)
      constraint_params = data.constraint_params
      model, solving_constraint = \
          models.construct_model(
            config = config,
            constraint_param = constraint_params
          )

      solver_ = solver.get_solver_cls(config.solver_type)(config=config)
      s = solver_(
          model,
          solving_constraint=solving_constraint,
          solving_params=constraint_params,
          ret_solution=True)
      v = utils.DimacsVerifier(test_file)
      self.assertTrue(v.is_correct(utils.z3_cnf_solution_str(s)))
