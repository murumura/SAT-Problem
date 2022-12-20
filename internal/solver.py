import abc
from internal import configs
from internal import models
from typing import Any, Optional, Tuple
import z3
from qiskit.circuit.library.phase_oracle import PhaseOracle
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram

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
  """A based solver containing operation for SAT/SMT/QC problem solving"""

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
      solving_params: dict, **kwargs):
    """Solve """


@register_solver('qiskit')
class QSSolver(BaseSolver):

  def print_solution(
      self, sol, solver_model, solving_params: dict, **kwargs) -> str:
    ret_sol = ""
    print(f"Solution of {self.puzzle_type} problem:")
    if self.puzzle_type in ("sudoku", "latin_square"):
      mapping_dict = solver_model.auxiliary
      grid = solver_model.get_vars()
      for i, (r, c, v) in mapping_dict.items():
        grid[r][c] = v if sol[i] == '1' else 0
      for i in grid:
        s_line = ' '.join(map(str, i))
        ret_sol += s_line + "\n"
    elif self.puzzle_type == "cnf":
      # bind boolean solution to orderdict
      res_dict = solver_model.get_vars()
      res_dict.update(
          (f"cnf_%s" % (i + 1), " = True" if val == '1' else " = False")
          for i, val in enumerate(sol))
      ret_sol = "[" + ' '.join(
          [
              ("," if i != 0 else '') + key + res_dict.get(key, '')
              for (i, key) in enumerate(res_dict.keys())
          ]) + "]"
    print(ret_sol)
    return ret_sol

  def __call__(
      self,
      model: models.Model,
      solving_constraint: Optional[Tuple[Any, ...]],
      solving_params: dict,
      ret_solution: bool = False,
      **kwargs):

    def prepare_default_grover(
        use_sampler: str, iterations=None, growth_rate=None):
      """Prepare Grover instance"""
      if use_sampler == "ideal":
        sampler = Sampler()
        grover = Grover(
            sampler=sampler, iterations=iterations, growth_rate=growth_rate)
      elif use_sampler == "shots":
        sampler_with_shots = Sampler(options={"shots": 1024, "seed": 123})
        grover = Grover(
            sampler=sampler_with_shots,
            iterations=iterations,
            growth_rate=growth_rate)
      else:
        raise ValueError(
            f"Unsupport use_sampler type, only supoort ideal, shots, got {use_sampler}"
        )
      return grover

    boolean_expr = solving_constraint
    oracle = PhaseOracle(boolean_expr)

    verified_fn = model.verified_fn if model.verified_fn else oracle.evaluate_bitstring
    # The oracle can now be used to create an Grover instance:
    problem = AmplificationProblem(oracle, is_good_state=verified_fn)
    result = None
    grover = prepare_default_grover(use_sampler="shots", iterations=None)

    ret_sol = None
    if problem is not None:
      result = grover.amplify(problem)
      for i, dist in enumerate(result.circuit_results):
        if ret_sol is not None:
          break
        keys, values = zip(*sorted(dist.items(), reverse=False))
        for k_i, key in enumerate(keys):
          if (verified_fn(key)):
            ret_sol = self.print_solution(
                sol=key, solver_model=model, solving_params=solving_params)
            break
    if ret_sol is None:
      print("Failed to solve")
    if ret_solution:
      return ret_sol


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
      ret_solution: bool = False,
      **kwargs):
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
      print("Failed to solve")
      if ret_solution:
        return None
