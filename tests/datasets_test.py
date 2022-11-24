"""Tests for datasets."""

from absl.testing import absltest
from internal import configs
from internal import datasets
import numpy as np

class DatasetsTest(absltest.TestCase):
  def test_load_from_dimacs_file(self):
    np.random.seed(0)
    dimacs_dict = datasets.load_cnf_from_dimacs("dataset/test2.dimacs")
    clauses_gt = np.array([[1,  2, -3], [-2, 3, 0]], dtype = np.int32)
    np.testing.assert_allclose(dimacs_dict['clauses'], clauses_gt, atol=1e-4, rtol=1e-4)
    self.assertEqual(2, dimacs_dict['num_clauses']) 
    self.assertEqual(3, dimacs_dict['num_vars']) 
    dimacs_dict = datasets.load_cnf_from_dimacs("dataset/test.dimacs")
    clauses_gt = np.array([
      [-1, -2, -3], 
      [1, -2, 3],
      [1, 2, -3],
      [1, -2, -3],
      [-1, 2, 3]
    ], dtype = np.int32)
    np.testing.assert_allclose(dimacs_dict['clauses'], clauses_gt, atol=1e-4, rtol=1e-4)
    self.assertEqual(5, dimacs_dict['num_clauses']) 
    self.assertEqual(3, dimacs_dict['num_vars']) 

  def test_load_puzzle_from_text(self):
    np.random.seed(0)
    puzzle_dict = datasets.load_cnf_from_puzzle("dataset/Latin_Square_Problem_HARD.txt")
    puzzle_gt = np.array([
      [0,1,2,0], 
      [2,0,0,1],
      [1,0,0,2],
      [0,0,1,0]
    ], dtype = np.int32)
    np.testing.assert_allclose(puzzle_dict['puzzle'], puzzle_gt, atol=1e-4, rtol=1e-4)
    self.assertEqual(4, puzzle_dict['num_rows']) 
    self.assertEqual(4, puzzle_dict['num_cols']) 

  def test_load_cnf_dataset(self):
    config = configs.Config(
        data_file = "dataset/test.dimacs",
        data_ext = "dimacs"
    )
    data = datasets.Dataset(config)
    constraint_params = data.constraint_params
    clauses_gt = np.array([
      [-1, -2, -3], 
      [1, -2, 3],
      [1, 2, -3],
      [1, -2, -3],
      [-1, 2, 3]
    ], dtype = np.int32)
    np.testing.assert_allclose(constraint_params['clauses'], clauses_gt, atol=1e-4, rtol=1e-4)
    self.assertEqual(5, constraint_params['num_clauses']) 
    self.assertEqual(3, constraint_params['num_vars'])

  def test_load_puzzle_dataset(self):
    config = configs.Config(
        data_file = "dataset/Latin_Square_Problem_HARD.txt",
        data_ext = "txt",
        puzzle_type = "sudoku",
        solver_type = "z3"
    )
    data = datasets.Dataset(config)
    constraint_params = data.constraint_params
    puzzle_gt = np.array([
      [0,1,2,0], 
      [2,0,0,1],
      [1,0,0,2],
      [0,0,1,0]
    ], dtype = np.int32)
    np.testing.assert_allclose(constraint_params['puzzle'], puzzle_gt, atol=1e-4, rtol=1e-4)
    self.assertEqual(4, constraint_params['num_rows']) 
    self.assertEqual(4, constraint_params['num_cols']) 