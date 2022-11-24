from typing import Any, Callable, Optional, Tuple
from absl import flags
from internal import utils
import dataclasses
import gin

@gin.configurable()
@dataclasses.dataclass
class Config:
  """Configuration flags for everything."""
  checkpoint_dir: Optional[str] = None  # Where to log checkpoints.
  data_file: Optional[str] = None  # Input data directory.
  data_ext: Optional[str] = 'dimacs'  # Data extension
  puzzle_type: str = 'sudoku'  # only cnf | sudoku | latin_square are supported
  solver_type: str = 'z3'

def define_common_flags():
  # Define the flags used by both train.py and eval.py
  flags.DEFINE_string('mode', None, 'Required by GINXM, not used.')
  flags.DEFINE_string('base_folder', None, 'Required by GINXM, not used.')
  flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
  flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')

def load_config(save_config=True):
  """Load the config, and optionally checkpoint it."""
  gin.parse_config_files_and_bindings(
      flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, skip_unknown=True)
  config = Config()
  if save_config:
    utils.makedirs(config.checkpoint_dir)
    with utils.open_file(config.checkpoint_dir + '/config.gin', 'w') as f:
      f.write(gin.config_str())
  return config
