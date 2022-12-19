"""Basic CNF form, Sudoku Puzzle and Latin Square Puzzle, with helper functions for problem construction and encoding."""
from internal import configs
from internal import utils
from typing import Tuple
from collections import OrderedDict
import itertools
import z3
import sympy.logic.boolalg as sympy_logic
import math
import qiskit
import sympy
# importing Qiskit
from qiskit import IBMQ, Aer, assemble, transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from sympy.logic import simplify_logic

constraint_registry = {}


def register_constraint(name):

  def register_constraint_fn(cls):
    if name in constraint_registry:
      raise ValueError('Cannot register duplicate constraint ({})'.format(name))
    constraint_registry[name] = cls
    return cls

  return register_constraint_fn


def get_constrain_fn(name):
  if name not in constraint_registry:
    print(constraint_registry)
    raise ValueError('Cannot find constraint fn {}'.format(name))
  return constraint_registry[name]


@register_constraint('z3_smt_sudoku')
def sudoku_smt_constraint(constraint_param: dict) -> dict:
  """Reduce sudoku problem to SMT form."""
  rows, cols = constraint_param['num_rows'], constraint_param['num_cols']
  grid = constraint_param['puzzle'].tolist()
  sqrt_rows, sqrt_cols = int(math.sqrt(rows)), int(math.sqrt(cols))
  vars = [[z3.Int(f'x_{i}_{j}') for i in range(rows)] for j in range(cols)]
  range_constraint = [z3.And(i >= 1, i <= rows) for i in itertools.chain(*vars)]
  instance_constraint = \
    [z3.If(grid[i][j] == 0, True, grid[i][j] == vars[i][j])
              for i in range(rows) for j in range(cols)]

  row_constraint = [z3.Distinct(i) for i in vars]
  column_constraint = [
      z3.Distinct(list(itertools.chain(*vars))[i::rows]) for i in range(rows)
  ]
  square_constraint = \
        [z3.Distinct([vars[j + l][i + k] for l in range(sqrt_rows) for k in range(sqrt_cols)])
            for j in range(0, rows, sqrt_rows) for i in range(0, cols, sqrt_cols)]

  sudoku_constraint = \
    range_constraint + instance_constraint + row_constraint + \
    column_constraint + square_constraint

  return {"constraint": sudoku_constraint, "constraint_vars": vars}


@register_constraint('z3_sat_sudoku')
def sudoku_sat_clauses(constraint_param: dict) -> dict:
  """Reduce sudoku problem to SAT form."""
  rows, cols = constraint_param['num_rows'], constraint_param['num_cols']
  grid = constraint_param['puzzle'].tolist()
  sqrt_rows, sqrt_cols = int(math.sqrt(rows)), int(math.sqrt(cols))
  vars = [
      [
          [z3.Bool(f'x_{r}_{c}_{v}')
           for v in range(1, rows + 1)]
          for c in range(cols)
      ]
      for r in range(rows)
  ]
  num_vars = rows
  c0 = [
      vars[r][c][grid[r][c] - 1]
      for r in range(rows)
      for c in range(cols)
      if grid[r][c] != 0
  ]
  c0 = z3.And(*c0)
  # each entry has at least one value
  c1 = [z3.Or([*vars[r][c]]) for r in range(rows) for c in range(cols)]
  c1 = z3.And(*c1)
  # each entry has at most one value
  c2 = \
    [z3.Or([z3.Not(vars[r][c][v]), z3.Not(vars[r][c][v_])])
      for r in range(rows) for c in range(cols) for v in range(num_vars) for v_ in range(v)]
  c2 = z3.And(*c2)
  # each row has all numbers
  c3 = [
      z3.Or([vars[r][c][v]
             for c in range(cols)])
      for v in range(num_vars)
      for r in range(rows)
  ]
  c3 = z3.And(*c3)
  # each columns has all numbers
  c4 = [
      z3.Or([vars[r][c][v]
             for r in range(rows)])
      for v in range(num_vars)
      for c in range(cols)
  ]
  c4 = z3.And(*c4)
  # each blocks has all numbers
  c5 = \
        [z3.Or([vars[j + l][i + k][v] for l in range(sqrt_rows) for k in range(sqrt_cols)])
            for j in range(0, rows, sqrt_rows) for i in range(0, cols, sqrt_cols) for v in range(num_vars)]
  c5 = z3.And(*c5)
  sudoku_constraint = z3.And(c1, c2, c3, c4, c5, c0)
  return {"constraint": sudoku_constraint, "constraint_vars": vars}


@register_constraint('z3_sat_latin_square')
def latin_square_sat_clauses(constraint_param: dict) -> dict:
  """Reduce latin square problem to SAT form."""
  rows, cols = constraint_param['num_rows'], constraint_param['num_cols']
  grid = constraint_param['puzzle'].tolist()
  vars = [
      [
          [z3.Bool(f'x_{r}_{c}_{v}')
           for v in range(1, rows + 1)]
          for c in range(cols)
      ]
      for r in range(rows)
  ]
  num_vars = rows
  c0 = [
      vars[r][c][grid[r][c] - 1]
      for r in range(rows)
      for c in range(cols)
      if grid[r][c] != 0
  ]
  c0 = z3.And(*c0)
  # each entry has at least one value
  c1 = [z3.Or([*vars[r][c]]) for r in range(rows) for c in range(cols)]
  c1 = z3.And(*c1)
  # each entry has at most one value
  c2 = \
    [z3.Or([z3.Not(vars[r][c][v]), z3.Not(vars[r][c][v_])])
      for r in range(rows) for c in range(cols) for v in range(num_vars) for v_ in range(v)]
  c2 = z3.And(*c2)
  # each row has all numbers
  c3 = [
      z3.Or([vars[r][c][v]
             for c in range(cols)])
      for v in range(num_vars)
      for r in range(rows)
  ]
  c3 = z3.And(*c3)
  # each columns has all numbers
  c4 = [
      z3.Or([vars[r][c][v]
             for r in range(rows)])
      for v in range(num_vars)
      for c in range(cols)
  ]
  c4 = z3.And(*c4)

  sudoku_constraint = z3.And(c1, c2, c3, c4, c0)
  return {"constraint": sudoku_constraint, "constraint_vars": vars}


@register_constraint('z3_smt_latin_square')
def latin_square_smt_constraint(constraint_param: dict) -> dict:
  """Reduce latin square problem to SMT form."""
  rows, cols = constraint_param['num_rows'], constraint_param['num_cols']
  grid = constraint_param['puzzle'].tolist()
  vars = [[z3.Int(f'x_{i}_{j}') for i in range(rows)] for j in range(cols)]
  range_constraint = [z3.And(i >= 1, i <= rows) for i in itertools.chain(*vars)]
  instance_constraint = \
    [z3.If(grid[i][j] == 0, True, grid[i][j] == vars[i][j])
              for i in range(rows) for j in range(cols)]
  row_constraint = [z3.Distinct(i) for i in vars]
  column_constraint = [
      z3.Distinct(list(itertools.chain(*vars))[i::rows]) for i in range(rows)
  ]
  latin_square_constraint = \
    range_constraint + instance_constraint + row_constraint + \
    column_constraint

  return {"constraint": latin_square_constraint, "constraint_vars": vars}


@register_constraint('qiskit_sat_cnf')
def qc_cnf_constraint(constraint_param: dict) -> dict:
  """ Create a BooleanExpression from the string in the DIMACS format."""
  # Encode the k-sat problem in to Boolean expression form
  clauses = constraint_param['clauses'].tolist()
  num_vars = constraint_param['num_vars']
  var_names = [f"cnf_%s" % (i + 1) for i in range(num_vars)]
  sympy_clauses = []
  for clause in clauses:
    list = []
    for lit in clause:
      num = int(lit)
      if num == 0:
        continue
      negative = (num < 0)
      if negative:
        list.append(~sympy.core.Symbol("cnf_%s" % abs(num)))
      else:
        list.append(sympy.core.Symbol("cnf_%s" % abs(num)))
    if len(list) > 0:
      sympy_clauses.append(sympy.logic.boolalg.Or(*list))
  boolean_expr = repr(simplify_logic(sympy.logic.boolalg.And(*sympy_clauses)))
  # construct a order dict for storing solving variable
  c_vars = OrderedDict([(name, None) for name in var_names])
  return {"constraint": boolean_expr, "constraint_vars": c_vars}


@register_constraint('qiskit_sat_sudoku')
def qc_sudoku_clauses(constraint_param: dict) -> dict:
  pass


@register_constraint('qiskit_sat_latin_square')
def qc_latin_square_clauses(constraint_param: dict) -> dict:
  rows, cols = constraint_param['num_rows'], constraint_param['num_cols']
  grid = constraint_param['puzzle'].tolist()
  if rows >= 4 or cols >= 4:
    raise ValueError(f"Only supports puzzle sizes <= 4x4, got {rows}x{cols}")
  num_vars = rows
  sympy_clauses = []
  sympy_vars = [
      [
          [sympy.core.Symbol(f'x_{r}_{c}_{v}')
           for v in range(1, rows + 1)]
          for c in range(cols)
      ]
      for r in range(rows)
  ]
  # consider unsolved variables
  c0 = [
      sympy_vars[r][c][grid[r][c] - 1]
      for r in range(rows)
      for c in range(cols)
      if grid[r][c] != 0
  ]
  c0 = sympy_logic.to_cnf(sympy_logic.And(*c0))
  sympy_clauses.append(c0)
  # each entry has at least one value
  c1 = [
      sympy_logic.Or(*[*sympy_vars[r][c]])
      for r in range(rows)
      for c in range(cols)
  ]
  c1 = sympy_logic.to_cnf(simplify_logic(sympy_logic.And(*c1)))
  sympy_clauses.append(c1)
  # each entry has at most one value
  c2 = \
    [sympy_logic.Or(*[sympy_logic.Not(sympy_vars[r][c][v]), sympy_logic.Not(sympy_vars[r][c][v_])])
      for r in range(rows) for c in range(cols) for v in range(num_vars) for v_ in range(v)]
  c2 = sympy_logic.to_cnf(simplify_logic(sympy_logic.And(*c2)))
  sympy_clauses.append(c2)
  # each row has all numbers
  c3 = [
      sympy_logic.Or(*[sympy_vars[r][c][v]
                       for c in range(cols)])
      for v in range(num_vars)
      for r in range(rows)
  ]
  c3 = sympy_logic.to_cnf(simplify_logic(sympy_logic.And(*c3)))
  sympy_clauses.append(c3)
  # each columns has all numbers
  c4 = [
      sympy_logic.Or(*[sympy_vars[r][c][v]
                       for r in range(rows)])
      for v in range(num_vars)
      for c in range(cols)
  ]
  c4 = sympy_logic.to_cnf(simplify_logic(sympy_logic.And(*c4)))
  sympy_clauses.append(c4)
  sympy_expr = \
      sympy_logic.to_cnf(
          simplify_logic(sympy.logic.boolalg.And(*sympy_clauses)))
  boolean_expr = repr(sympy_expr)
  dimacs_f = utils.to_dimacs_formula(sympy_expr)
  print(dimacs_f)
  return {"constraint": boolean_expr, "constraint_vars": None}


@register_constraint('z3_sat_cnf')
def cnf_z3_constraint(constraint_param: dict) -> dict:
  """Reduce CNF expression to Z3 solvable form."""

  def z3_sign(val, z3_var):
    if (val < 0):
      return z3.Not(z3_var)
    return z3_var

  clauses = constraint_param['clauses'].tolist()
  num_vars = constraint_param['num_vars']
  num_clauses = constraint_param['num_clauses']
  vars = [z3.Bool(f'x_{i}') for i in range(num_vars)]
  c_list = []
  for n_c in range(num_clauses):
    caluse_c = [
        z3.Or(
            [
                z3_sign(clauses[n_c][i], vars[i])
                for i in range(num_vars)
                if clauses[n_c][i] != 0
            ])
    ]
    c_list.append(caluse_c)
  # flatten list
  c_list = [element for sublist in c_list for element in sublist]
  cnf_constraint = z3.And(*c_list)

  return {"constraint": cnf_constraint, "constraint_vars": vars}


class Model(object):

  def __init__(self, config: configs.Config):
    super().__init__()
    self.config = config
    self.vars = None

  def init(self, constraint_param: dict) -> dict:
    """Generate constraint passed to corresponding solver."""
    constraint_key = self.config.solver_type + "_" + self.config.problem_type + "_" + self.config.puzzle_type
    constraint_fn = get_constrain_fn(constraint_key)
    constraint_dict = constraint_fn(constraint_param)
    self.vars = constraint_dict['constraint_vars']
    return constraint_dict['constraint']

  def get_vars(self):
    """Return solvable variables"""
    return self.vars


def construct_model(constraint_param: dict,
                    config: configs.Config) -> Tuple[Model, dict]:
  """Construct a SAT/SMT/QC based model.
    Args:
      constraint_param: A dict consists of parameters for generating solving constraint.
    Returns:
      model: initialized sat/smt based model with parameters.
      solving_constraint: dictonary that contains related constraint within solvable variable to specified solver.
  """
  model = Model(config=config)
  solving_constraint = model.init(constraint_param)
  return model, solving_constraint
