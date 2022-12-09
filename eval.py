"""Evaluation script."""
from absl import app
import gin
from internal import configs
from internal import datasets
from internal import solver
from internal import models

configs.define_common_flags()


def main(unused_argv):
  config = configs.load_config(save_config=True)
  data = datasets.Dataset(config)
  model, solving_constraint = \
        models.construct_model(
          config = config,
          constraint_param = data.constraint_params
        )
  cfg_solver = solver.get_solver_cls(config.solver_type)(config=config)

  cfg_solver(
      model,
      solving_constraint=solving_constraint,
      solving_params=data.constraint_params)


if __name__ == '__main__':
  with gin.config_scope('eval'):
    app.run(main)
