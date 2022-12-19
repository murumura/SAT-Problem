PROBLEM_FILE=test.dimacs
EXPERIMENT=cnf
SOLVER=z3
PROBLEM_TYPE=sat
DATA_DIR=./datasets
CHECKPOINT_DIR=${PWD}/checkpoint/"$EXPERIMENT"/"$PROBLEM_TYPE"/"$SOLVER"

python -m eval \
  --gin_configs=./configs/cnf_sat_z3.gin \
  --gin_bindings="Config.data_file = '${DATA_DIR}/${PROBLEM_FILE}'" \
  --gin_bindings="Config.checkpoint_dir = '${CHECKPOINT_DIR}'"
