"""Utility functions."""
import enum
import os
import re
import math
import itertools
import numpy as np
from PIL import Image
from typing import List
from sympy.core.symbol import Symbol
from sympy.logic.boolalg import to_cnf, And, Or, Not


class DimacsVerifier():
  """Create an object that can be used to check whether
    an assignment satisfies a DIMACS file.
        Args:
            dimacs_file (str): path to the DIMACS file
    """

  def __init__(self, dimacs_file):
    with open(dimacs_file, 'r') as f:
      self.dimacs = f.read()

  def is_correct(self, guess):
    """Verifies a SAT solution against this object's
        DIMACS file.
        Args:
            guess (str): Assignment to be verified.
                          Must be string of 1s and 0s.
        Returns:
            bool: True if `guess` satisfies the
                        problem. False otherwise.
    """
    # Convert characters to bools
    guess = [bool(int(x)) for x in guess]
    for line in self.dimacs.split('\n'):
      if not line:  # ignore empty line
        continue
      line = line.lstrip().rstrip("0").rstrip()
      clause_eval = False
      for literal in re.split(r'\s+', line):
        if literal in ['p', 'c']:
          # line is not a clause
          clause_eval = True
          break
        if '-' in literal:
          literal = literal.strip('-')
          lit_eval = not guess[int(literal) - 1]
        else:
          lit_eval = guess[int(literal) - 1]
        clause_eval |= lit_eval
      if clause_eval is False:
        return False
    return True


def latin_square_verify(grid: List[List[int]], n_rows: int):
  """Verifies a given grid forms a valid latin square solution.
      Args:
          grid (list): 2-dimentional list describe placement numbers onto the puzzle.
          n_rows (int): number of rows of the puzzle.
      Returns:
          bool: True if placement forms a valid latin square solution.
            False otherwise.
  """

  def lsq_ok(line):
    return (len(line) == n_rows and sum(line) == sum(set(line)))

  bad_rows = [row for row in grid if not lsq_ok(row)]
  grid = list(zip(*grid))
  bad_cols = [col for col in grid if not lsq_ok(col)]
  return not (bad_rows or bad_cols)


def sudoku_verify(grid: List[List[int]], n_rows: int, n_cols: int):
  """Verifies a given grid forms a valid sudoku solution.
      Args:
          grid (list): 2-dimentional list describe placement numbers onto the puzzle.
          n_rows (int): number of rows of the puzzle.
          n_cols (int): number of column of the puzzle.
      Returns:
          bool: True if placement forms a valid sudoku solution.
            False otherwise.
  """
  sqrt_rows, sqrt_cols = int(math.sqrt(n_rows)), int(math.sqrt(n_cols))

  def sudoku_ok(line):
    return (len(line) == n_rows and sum(line) == sum(set(line)))

  bad_rows = [row for row in grid if not sudoku_ok(row)]
  grid = list(zip(*grid))
  bad_cols = [col for col in grid if not sudoku_ok(col)]
  squares = []
  for i in range(0, n_rows, sqrt_rows):
    for j in range(0, n_cols, sqrt_cols):
      square = list(
          itertools.chain(
              row[j:j + sqrt_cols] for row in grid[i:i + sqrt_rows]))
      square = [element for sublist in square for element in sublist]
      squares.append(square)
  bad_squares = [square for square in squares if not sudoku_ok(square)]
  return not (bad_rows or bad_cols or bad_squares)


class DimacsMapping:

  def __init__(self):
    self._symbol_to_variable = {}
    self._variable_to_symbol = {}
    self._total_variables = 0

  @property
  def total_variables(self):
    return self._total_variables

  def new_variable(self):
    self._total_variables += 1
    return self._total_variables

  def get_variable_for(self, symbol):
    result = self._symbol_to_variable.get(symbol)
    if result is None:
      result = self.new_variable()
      self._symbol_to_variable[symbol] = result
      self._variable_to_symbol[result] = symbol

    return result

  def get_symbol_for(self, variable):
    return self._variable_to_symbol[variable]

  def __str__(self) -> str:
    return str(self._variable_to_symbol)


def to_dimacs_formula(sympy_cnf):
  dimacs_mapping = DimacsMapping()
  dimacs_clauses = []

  assert type(sympy_cnf) == And
  for sympy_clause in sympy_cnf.args:
    assert type(sympy_clause) == Or

    dimacs_clause = []
    for sympy_literal in sympy_clause.args:
      if type(sympy_literal) == Not:
        sympy_symbol, polarity = sympy_literal.args[0], -1
      elif type(sympy_literal) == Symbol:
        sympy_symbol, polarity = sympy_literal, 1
      else:
        raise AssertionError("invalid cnf")

      dimacs_variable = dimacs_mapping.get_variable_for(sympy_symbol)
      dimacs_literal = dimacs_variable * polarity
      dimacs_clause.append(dimacs_literal)

    dimacs_clauses.append(dimacs_clause)

  return DimacsFormula(dimacs_mapping, dimacs_clauses)


class DimacsFormula:

  def __init__(self, mapping, clauses):
    self._mapping = mapping
    self._clauses = clauses

  @property
  def mapping(self):
    return self._mapping

  @property
  def clauses(self):
    return self._clauses

  def __str__(self):
    header = f"p cnf {self._mapping.total_variables} {len(self._clauses)}"
    body = "\n".join(
        " ".join([str(literal)
                  for literal in clause] + ["0"])
        for clause in self._clauses)

    return "\n".join([header, body])


class DataExtension(enum.Enum):
  """Dataset split."""
  TEXT = 'txt'
  DIMACS = 'dimacs'


def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def file_exists(pth):
  return os.path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return os.path.isdir(pth)


def makedirs(pth):
  if not file_exists(pth):
    os.makedirs(pth)


def get_extention(pth):
  return os.path.splitext(pth)[1][1:]


def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open_file(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)
  return image


def puzzle_sol_str_to_list(grid: str, n_rows: int,
                           n_cols: int) -> List[List[int]]:
  """Convert a puzzle solution from string to list of list."""
  x = grid.split()
  x = [[int(x[j + i * n_cols]) for j in range(n_cols)] for i in range(n_rows)]
  return x


def z3_cnf_solution_str(z3_cnf_sol: str) -> str:
  """Convert a z3.Boolean CNF solution to string of 1s and 0s with numerical order of variable suffix.
  Eg:
    [x_2 = True, x_0 = True,  x_1 = False] =>  101
    [x_2 = True, x_0 = False, x_1 = False] =>  001
  """
  # yapf: disable
  asgmt_patterns = [
      ("[A-Za-z0-9]+_[0-9]+[\s]*=[\s]*True", "1"),
      ("[A-Za-z0-9]+_[0-9]+[\s]*=[\s]*False", "0")
  ]
  suffix_pattern = re.compile('[0-9]+')
  # yapf: enable
  finder = [(re.compile(p), s) for p, s in asgmt_patterns]
  asgmt_group = re.findall(
      r'[A-Za-z0-9]+_[0-9]+[\s]*=[\s]*True|[A-Za-z0-9]+_[0-9]+[\s]*=[\s]*False',
      z3_cnf_sol)
  asgmt_vals = [
      re.sub(p, s, asgmt)
      for asgmt in asgmt_group
      for p, s in finder
      if re.match(p, asgmt)
  ]
  suffix = [
      int(suffix_pattern.search(x).group(0))
      for x in re.findall(r'[A-Za-z0-9]+_[0-9]+', z3_cnf_sol)
  ]
  sol_str = [x for _, x in sorted(zip(suffix, asgmt_vals))]
  return ''.join(sol_str)
