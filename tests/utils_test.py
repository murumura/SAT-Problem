"""Tests for utils."""
from absl.testing import absltest
from internal import utils


class UtilsTest(absltest.TestCase):

  def test_z3_cnf_solution_string(self):
    self.assertEqual(
        "101",
        utils.z3_cnf_solution_str("[x_2 = True, x_0 = True,  x_1 = False]"))
    self.assertEqual(
        "111",
        utils.z3_cnf_solution_str("[x_2 = True, x_0 = True,  x_1 = True]"))
    self.assertEqual(
        "001",
        utils.z3_cnf_solution_str("[x_2 = True, x_0 = False,  x_1 = False]"))
    self.assertEqual(
        "100",
        utils.z3_cnf_solution_str("[x_2 = False, x_0 = True,  x_1 = False]"))
    self.assertEqual(
        "101100",
        utils.z3_cnf_solution_str(
            "[x_5= False, x_4 =False,  x_3=True,x_2 = True, x_1 = False,  x_0=True]"
        ))
