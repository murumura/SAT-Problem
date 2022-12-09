import abc
from internal import configs
from internal import models
from typing import Any, Optional, Tuple
import z3

solver_registry = {}


def register_solver(name):

  def register_solver_cls(cls):
    if name in solver_registry:
      raise ValueError('Cannot register duplicate solver ({})'.format(name))
    solver_registry[name] = cls
    return cls

  return register_solver_cls


def get_solver_cls(name):
  if name not in solver_registry:
    print(solver_registry)
    raise ValueError('Cannot find solver fn {}'.format(name))
  return solver_registry[name]


class BaseSolver(metaclass=abc.ABCMeta):
  """A based solver containing operation for SAT/SMT problem solving"""

  def __init__(self, config: configs.Config):
    super().__init__()
    self.solver_type = config.solver_type
    self.puzzle_type = config.puzzle_type
    self.problem_type = config.problem_type

  @abc.abstractmethod
  def print_solution(self, **kwargs):
    """Print out solution by user-specified config parameters and return solution as string."""

  @abc.abstractmethod
  def __call__(
      self, model: models.Model, solving_constraint: Optional[Tuple[Any, ...]],
      solving_params: dict):
    """Solve """


@register_solver('z3')
class Z3Solver(BaseSolver):

  def print_solution(self, solver_model, vars, solving_params) -> str:
    ret_sol = ""
    print(f"Solution of {self.puzzle_type} problem:")
    if self.puzzle_type in ("sudoku", "latin_square"):
      n_rows = solving_params['num_rows']
      n_cols = solving_params['num_cols']
      if self.problem_type == "smt":
        r = [
            [solver_model.evaluate(vars[j][i])
             for i in range(n_rows)]
            for j in range(n_cols)
        ]
        for i in r:
          s_line = ' '.join(map(str, i))
          ret_sol += s_line + "\n"
      elif self.problem_type == "sat":
        r = [
            [
                [
                    v + 1
                    for v in range(n_rows)
                    if solver_model.evaluate(vars[r][c][v])
                ]
                for c in range(n_cols)
            ]
            for r in range(n_rows)
        ]
        for i in r:
          s_line = ' '.join(map(str, i)).replace('[', '').replace(']', '')
          ret_sol += s_line + "\n"
    elif self.puzzle_type == "cnf":
      ret_sol = str(solver_model)
    print(ret_sol)
    return ret_sol

  def __call__(
      self,
      model: models.Model,
      solving_constraint: Optional[Tuple[Any, ...]],
      solving_params: dict,
      ret_solution: bool = False):
    z3_solver = z3.Solver()
    z3_solver.add(solving_constraint)
    if z3_solver.check() == z3.sat:
      s = self.print_solution(
          solver_model=z3_solver.model(),
          vars=model.get_vars(),
          solving_params=solving_params)
      if ret_solution:
        return s
    else:
      print("failed to solve")
