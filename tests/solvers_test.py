"""Tests for solvers."""

from internal import models
from internal import datasets
from internal import configs
from internal import solver
from internal import utils
from absl.testing import absltest
import numpy as np
from sympy.logic.utilities.dimacs import load
from sympy.logic import simplify_logic

def sympy_load_file(location):
  """Loads a boolean expression from a file."""
  with open(location) as f:
    s = f.read()
  return load(s)


class SolversTest(absltest.TestCase):

  def test_sudoku_solving(self):
    np.random.seed(0)
    for test_file in ("dataset/puzzle2.txt", "dataset/puzzle.txt",
                      "dataset/puzzle3.txt"):
      for prob_type in ("sat", "smt"):
        config = configs.Config(
            data_file=test_file,
            data_ext="txt",
            puzzle_type="sudoku",
            solver_type="z3",
            problem_type=prob_type)
        data = datasets.Dataset(config)
        constraint_params = data.constraint_params
        model, solving_constraint = \
            models.construct_model(
              config = config,
              constraint_param = constraint_params
            )

        cfg_solver = solver.get_solver_cls(config.solver_type)(config=config)
        rows, cols = constraint_params['num_rows'], constraint_params[
            'num_cols']
        s = cfg_solver(
            model,
            solving_constraint=solving_constraint,
            solving_params=constraint_params,
            ret_solution=True)
        sudoku_sol = utils.puzzle_sol_str_to_list(s, n_rows=rows, n_cols=cols)

        self.assertTrue(
            utils.sudoku_verify(sudoku_sol, n_rows=rows, n_cols=cols))

  def test_latin_square_solving(self):
    np.random.seed(0)
    for test_file in ("dataset/Latin_Square_Problem_EASY.txt",
                      "dataset/Latin_Square_Problem_MEDIUM.txt",
                      "dataset/Latin_Square_Problem_HARD.txt",
                      "dataset/puzzle.txt"):
      for prob_type in ("sat", "smt"):
        config = configs.Config(
            data_file=test_file,
            data_ext="txt",
            puzzle_type="latin_square",
            solver_type="z3",
            problem_type=prob_type)
        data = datasets.Dataset(config)
        constraint_params = data.constraint_params
        model, solving_constraint = \
            models.construct_model(
              config = config,
              constraint_param = constraint_params
            )
        rows, cols = constraint_params['num_rows'], constraint_params[
            'num_cols']
        cfg_solver = solver.get_solver_cls(config.solver_type)(config=config)
        s = cfg_solver(
            model,
            solving_constraint=solving_constraint,
            solving_params=constraint_params,
            ret_solution=True)
        ls_sol = utils.puzzle_sol_str_to_list(s, n_rows=rows, n_cols=cols)
        self.assertTrue(utils.latin_square_verify(ls_sol, n_rows=rows))

  def test_cnf_solving(self):
    np.random.seed(0)
    for test_file in ("dataset/test.dimacs", "dataset/test2.dimacs",
                      "dataset/test3.dimacs", "dataset/test4.dimacs",
                      "dataset/test5.dimacs"):
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

      cfg_solver = solver.get_solver_cls(config.solver_type)(config=config)
      s = cfg_solver(
          model,
          solving_constraint=solving_constraint,
          solving_params=constraint_params,
          ret_solution=True)
      v = utils.DimacsVerifier(test_file)
      self.assertTrue(v.is_correct(utils.z3_cnf_solution_str(s)))

  def test_qc_cnf_solving(self):
    np.random.seed(0)
    for test_file in ("dataset/test.dimacs", "dataset/test2.dimacs",
                      "dataset/4sat.dimacs", "dataset/6sat.dimacs"):
      config = configs.Config(
          data_file=test_file,
          data_ext="dimacs",
          puzzle_type="cnf",
          solver_type="qiskit",
          problem_type="sat")
      data = datasets.Dataset(config)
      constraint_params = data.constraint_params
      model, solving_constraint = \
          models.construct_model(
            config = config,
            constraint_param = constraint_params
          )

      cfg_solver = solver.get_solver_cls(config.solver_type)(config=config)
      s = cfg_solver(
          model,
          solving_constraint=solving_constraint,
          solving_params=constraint_params,
          ret_solution=True)
      v = utils.DimacsVerifier(test_file)
      self.assertTrue(v.is_correct(utils.z3_cnf_solution_str(s)))

  def test_cnf_boolean_expr(self):
    np.random.seed(0)
    for test_file in ("dataset/test.dimacs", "dataset/test2.dimacs",
                      "dataset/test3.dimacs", "dataset/test4.dimacs",
                      "dataset/test5.dimacs"):
      config = configs.Config(
          data_file=test_file,
          data_ext="dimacs",
          puzzle_type="cnf",
          solver_type="qiskit",
          problem_type="sat")
      data = datasets.Dataset(config)
      constraint_params = data.constraint_params
      boolean_expr = models.qc_cnf_constraint(constraint_params)['constraint']
      self.assertEqual(boolean_expr, repr(simplify_logic(sympy_load_file(test_file))))
