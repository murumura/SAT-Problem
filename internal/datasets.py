import string
from internal import configs
from internal import utils
import threading
import numpy as np


def load_cnf_from_dimacs(dimacs_file: str = None) -> dict:
  """Parsing and forming set of correspoding CNF caluses from DIMACS fiie.
     Args:
      dimacs_file (str): path to the DIMACS file
     Returns:
      Dictionary that contains following entries:
       clauses (np.ndarray): array of parsed clauses of each literal, free varaibles are denoted as 0 if can't be ignored,
         for example line in DIMACS as "-x1, x3" would be parsed as "-1, 0, 3" in stored array, where x2 would be treated as
         free variable and hence been parsed as 0.
       num_clauses (int): number of clauses in DIMACS file.
       num_vars (int): total number of used variables in DIMACS file.
  """

  def spilt_dimacs_line(s: str) -> list:
    result = []
    was_in_token = False
    token_start = None
    for i in range(len(s)):
      is_in_token = s[i] not in string.whitespace
      if not was_in_token and is_in_token:
        token_start = i
      elif was_in_token and not is_in_token:
        result.append((s[token_start:i], (token_start, i)))
      was_in_token = is_in_token
    if was_in_token:
      result.append((s[token_start], (token_start, len(s))))
    return result

  parsed_line_no = 0
  clause_list = []
  seen_problem_statement = False
  num_clauses = len(clause_list)
  num_vars = 0
  seen_clause = False
  max_var = 0
  if dimacs_file is not None:
    file = open(dimacs_file, 'r')
    for line in file:
      if (len(line) > 0):
        if line[
            0] == 'c':  # begins with the character c is considered a comment.
          pass
        elif not seen_problem_statement and line[0] == 'p':  # parse header line
          parsed_tokens = spilt_dimacs_line(line)
          if len(parsed_tokens) != 4:
            raise SyntaxError(
                f"invalid syntax: header line should be of the form p cnf <variables> <clause_list>, \
                got {len(parsed_tokens)} entries at line number: {parsed_line_no}"
            )
          elif parsed_tokens[0][0] != 'p':
            raise SyntaxError(
                f"invalid syntax: header line should be of the form p cnf <variables> <clause_list>, \
                got {parsed_tokens[0][0]} at first entry at line number: {parsed_line_no}"
            )
          elif parsed_tokens[1][0] != 'cnf':
            raise SyntaxError(
                f"invalid syntax: header line should be of the form p cnf <variables> <clause_list>, \
                got {parsed_tokens[1][0]} at first entry at line number: {parsed_line_no}"
            )
          else:
            try:
              num_vars = int(parsed_tokens[2][0])
              try:
                num_clauses = int(parsed_tokens[3][0])
              except ValueError:
                print(
                    f"Expect number of clause_list be integer, got {parsed_tokens[3][0]} at line number: {parsed_line_no}"
                )
            except ValueError:
              print(
                  f"Expect number of variables be integer, got {parsed_tokens[2][0]} at line number: {parsed_line_no}"
              )
        else:  # parse literals
          parsed_tokens = spilt_dimacs_line(line)
          for token, (col_start, col_end) in parsed_tokens:
            try:
              literal = int(token)
              if literal == 0:  # end of token
                if seen_clause:
                  seen_clause = False
              else:  # literal != 0
                if not seen_clause:
                  clause_list.append([])  # process start of clause
                  seen_clause = True
                clause_list[-1].append(literal)
                abs_literal = abs(literal)
                if abs_literal > max_var:
                  max_var = abs_literal
            except ValueError:
              print(
                  f"Expect literal be integer, got {token}, line no: {parsed_line_no}"
              )
        parsed_line_no += 1
    file.close()
    # sanity check before return
    if num_vars < max_var:
      raise ValueError(
          f"the declared number of variables {num_vars} is smaller than the actual number of variables {max_var}"
      )

    if num_clauses != len(clause_list):
      raise ValueError(
          f"the declared number of clause_list {num_clauses} does not match the actual number of clause_list {len(clause_list)}"
      )

  clauses_np = np.zeros(
      [len(clause_list),
       len(max(clause_list, key=lambda x: len(x)))],
      dtype=np.int32)

  for i, j in enumerate(clause_list):
    clauses_np[i][0:len(j)] = j

  return {
      "clauses": clauses_np,
      "num_clauses": int(num_clauses),
      "num_vars": int(num_vars)
  }


def load_cnf_from_puzzle(text_file: str = None) -> dict:
  """Parsing and forming set of correspoding CNF caluses from sudoku/latin square puzzle fiie.
     A puzzle file should be formated as follows: zero values indicating empty entries that should be solved later, non-zero values
     denote filling number, grids are seperated by comma.
     Args:
      text_file (str): path to the puzzle description file.
     Returns:
      Dictionary that contains following entries:
       puzzle (np.ndarray): array of parsed grid.
       num_rows (int): number of rows of specified grid.
       num_cols (int): number of columns of specified grid.
  """

  def extract_integer_line(s: str) -> list:
    result = []
    was_in_token = False
    token_start = None
    for i in range(len(s)):
      is_in_token = s[i] not in string.whitespace and s[i] not in ','
      if not was_in_token and is_in_token:
        token_start = i
      elif was_in_token and not is_in_token:
        result.append((s[token_start:i], (token_start, i)))
      was_in_token = is_in_token
    if was_in_token:
      result.append((s[token_start], (token_start, len(s))))
    return result

  clause_list = []
  parsed_line_no = 0
  if text_file is not None:
    file = open(text_file, 'r')
    for line in file:
      if (len(line) > 0):
        clause_list.append([])  # process start of clause
        parsed_string = extract_integer_line(line)
        for val_str, (col_start, col_end) in parsed_string:
          try:
            value = int(val_str)
            clause_list[-1].append(value)
          except ValueError:
            print(
                f"Expect literal be integer, got {val_str}, line no: {parsed_line_no}"
            )
        parsed_line_no += 1
    file.close()

  puzzle_grid = np.zeros(
      [len(clause_list),
       len(max(clause_list, key=lambda x: len(x)))],
      dtype=np.int32)

  for i, j in enumerate(clause_list):
    puzzle_grid[i][0:len(j)] = j
  num_rows, num_cols = puzzle_grid.shape
  return {
      "puzzle": puzzle_grid,
      "num_rows": int(num_rows),
      "num_cols": int(num_cols)
  }


class Dataset(threading.Thread):
  """Dataset class for constraint problem abstraction"""

  def __init__(self, config: configs.Config):
    super().__init__()
    # Initialize attributes
    self.file_ext = utils.DataExtension(config.data_ext)
    if not utils.file_exists(config.data_file):
      raise ValueError(f'Dataset file {config.data_file} does not exist.')
    self.data_file = config.data_file
    if self.file_ext == utils.DataExtension.DIMACS:
      self.load_fn = load_cnf_from_dimacs
    else:
      self.load_fn = load_cnf_from_puzzle

    self.start()

  @property
  def constraint_params(self):
    """Dict contains necessary parameters for building solvable constraint"""
    return self.load_fn(self.data_file)
