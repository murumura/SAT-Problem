"""Basic CNF form, Sudoku Puzzle and Latin Square Puzzle, with helper functions for construction and encoding."""
from internal import configs
import itertools
import z3
import math

constrain_registry = {}

def register_constrain(name):
  def register_constrain_fn(cls):
    if name in constrain_registry:
      raise ValueError('Cannot register duplicate constrain ({})'.format(name))
    constrain_registry[name] = cls
    return cls
  return register_constrain_fn

def get_constrain_fn(name):
  if name not in constrain_registry:
    print(constrain_registry)
    raise ValueError('Cannot find constrain fn {}'.format(name))
  return constrain_registry[name]

@register_constrain('z3_sudoku')
def sudoku_z3_constraint(constraint_param: dict) -> dict:
  rows, cols = constraint_param['num_rows'], constraint_param['num_cols']
  grid = constraint_param['puzzle'].tolist()
  sqrt_rows, sqrt_cols = int(math.sqrt(rows)), int(math.sqrt(cols))
  vars = [[z3.Int(f'x_{i}_{j}') for i in range(rows)] for j in range(cols)]
  range_constraint= [z3.And(i >= 1, i <= rows) for i in itertools.chain(*vars)]
  instance_constraint = \
    [z3.If(grid[i][j] == 0, True, grid[i][j] == vars[i][j])
              for i in range(rows) for j in range(cols)]

  row_constraint    = [z3.Distinct(i) for i in vars]
  column_constraint = [z3.Distinct(list(itertools.chain(*vars))[i::rows]) for i in range(rows)]
  square_constraint = \
        [z3.Distinct([vars[j + l][i + k] for l in range(sqrt_rows) for k in range(sqrt_cols)])
            for j in range(0, rows, sqrt_rows) for i in range(0, cols, sqrt_cols)]

  sudoku_constraint = \
    range_constraint + instance_constraint + row_constraint + \
    column_constraint + square_constraint

  return {
    "constraint" : sudoku_constraint, 
    "constraint_vars" : vars
  }

@register_constrain('z3_cnf')
def cnf_z3_constraint(constraint_param: dict) -> dict:
  def z3_neg(val, z3_var):
    if (val < 0):
      return z3.Not(z3_var)
    return z3_var
  clauses       = constraint_param['clauses'].tolist()
  num_vars      = constraint_param['num_vars']
  num_clauses   = constraint_param['num_clauses']
  vars          = [z3.Bool(f'x_{i}') for i in range(num_vars)]
  c_list        = []
  for n_c in range(num_clauses):
    caluse_c = [z3.Or([z3_neg(clauses[n_c][i], vars[i]) for i in range(num_vars) if clauses[n_c][i] != 0])]
    c_list.append(caluse_c)
  # flatten list
  c_list = [element for sublist in c_list for element in sublist]
  cnf_constraint= z3.And(*c_list)
  
  return {
    "constraint" : cnf_constraint, 
    "constraint_vars" : vars
  }

class Model(object):
  def __init__(self, config: configs.Config):
    super().__init__()
    self.config = config
    self.vars = None
    
  def init(self, constraint_param) -> dict:
    """Generate constrint passed to corresponding solver."""
    constrain_key = self.config.solver_type + "_" + self.config.puzzle_type
    constrain_fn = get_constrain_fn(constrain_key)
    constraint_dict = constrain_fn(constraint_param)
    self.vars = constraint_dict['constraint_vars']
    return constraint_dict['constraint']

  def get_vars(self):
    """Return solvable variables"""
    return self.vars

def construct_model(constraint_param:dict, config:configs.Config):
  """Construct a SAT/SMT based model.
  Args:
    constraint_param: A dict consists of class.
  Returns:
    model: initialized sat/smt based model with parameters.
    solving_constraint: 
  """
  model = Model(config=config)
  solving_constraint = model.init(constraint_param)
  return model, solving_constraint

