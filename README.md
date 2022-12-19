# SAT/SMT-Problem
Solving CNF &amp; Latin-Square &amp; Sudoku problem by Z3-Solver and Qiskit

# Installation
## Option: Using pip
```bash
cd SAT-Problem
pip install -r requirements.txt
```

# Quick Start 

## Google Colab
I have provided corresponding jupyter notebooks for all problems, you can choose to execute on colab or locally, it is worth noting that if you choose to execute locally, you need to install the required packages in advance.

Below, the solver types that I have used to solve various problems types correspond to the relevant notebooks:
* `SAT_QC_CNF.ipynb` : solving SAT problem through quantum circuit powered by quskit. 
* `SAT_QC_Latin_Square.ipynb` : solving latin square puzzles through quantum circuit powered by quskit. 
* `SAT_QC_Sudoku.ipynb` : solving sudoku puzzles through quantum circuit powered by quskit. 
* `SAT_Z3_CNF.ipynb` : solving SAT problem through Z3 solver. 
* `SAT_Z3_Latin_Square.ipynb` : solving latin square  puzzles through modeling it to a SAT problem and solve it by Z3 solver. 
* `SAT_Z3_Sudoku.ipynb` : solving sudoku puzzles through modeling it to a SAT problem and solve it by Z3 solver.
* `SMT_Z3_Latin_Square.ipynb` : solving latin square  puzzles through modeling it to a SMT problem and solve it by Z3 solver. 
* `SMT_Z3_Sudoku.ipynb` : solving sudoku puzzles through modeling it to a SMT problem and solve it by Z3 solver.
## Scripts
Example scripts for evaluating all kinds of problems through different solving techniques can be found in `scripts/`. You'll need to change the paths to point to wherever the datasets are located. Gin configuration files for the problem model and some ablations can be found in `configs/`. 

## Details of the inner workings of Dataset
All files used as problem descriptions are placed in the datasets folder. The file with the extension dimacs is used to describe the cnf problem. The file with the extension txt represents the initial puzzle setting of latin square / sudoku. 

If you want to add other The file name of the problem file must be written according to the above description.