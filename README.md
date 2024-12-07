# 2D Cutting Stock Problem Solver

This repository provides an implementation of an exact method to solve the **2D Cutting Stock Problem (CSP)** using Mixed-Integer Programming (MIP). The goal is to minimize material waste by efficiently placing rectangular items on stock sheets while satisfying all placement and demand constraints.

## Features
- **Exact Solver**: Implements a Mixed-Integer Linear Programming (MILP) approach using the [COPT](https://www.copt.ai/) solver.
- **Visualization**: Generates graphical layouts of the cutting patterns for better insight into the solution.
- **Flexible Input Handling**:
  - Supports input files specifying stock and product dimensions.
  - Can generate random test data for quick experimentation.
- **Output Storage**: Saves results and visualizations in a designated output folder.

## Requirements
To use this repository, you will need:
- Python 3.8+
- Required Python packages:
  - `coptpy`
  - `numpy`
  - `matplotlib`
  - `argparse`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sliderboo/math-modeling.git
   cd math-modeling/exact_copt_solver
   ```
2. Install the required Python packages mentioned above.
   (Ensure that `coptpy` is installed and properly licensed.)

## Usage
The solver can be run with either custom input files or randomly generated data.

### Command-Line Arguments
- `-r`: Generate random stock and product dimensions.
- `-m`: Number of stock sheets to generate (default: 5).
- `-n`: Number of rectangular items to generate (default: 10).
- `-f`: Path to an input file containing stock and product dimensions.
- `-o`: Output folder to save results (default: `output`).

### Examples
1. **Run with Random Data**:
   ```bash
   python solver.py -r -m 5 -n 10 -o output_folder
   ```

2. **Run with Input File**:
   Prepare an input file, e.g., `input.txt`:
   ```python
   init_stocks = [(30, 40, 2), (50, 60, 1)]
   init_prods = [(10, 15, 4), (20, 10, 3), (25, 30, 2)]
   ```
   Run the solver:
   ```bash
   python solver.py -f input.txt -o output_folder
   ```

## Output
- **Initial testcase**: An `input.txt` file of the input
- **Visualization**: A `visualization.png` file illustrating the cutting layout for each stock sheet.
- **Details**: A log of material usage, waste, and fill percentage.

## Implementation Details
The solver leverages COPT for MIP optimization and includes:
- **Decision Variables**: Positions, orientations, material assignments, and non-overlap conditions for each item.
- **Constraints**: Ensure valid placement, non-overlapping items, and satisfaction of demand.
- **Objective Function**: Minimize the total material used or waste area.

For detailed information, refer to the `Implementation` section in the documentation.
[2D Cutting Stock Problem Report](https://www.papeeria.com/p/189cdd6765138e330b84adc9389d36e6#/report.tex)

## License
This project is licensed under the MIT License. See `LICENSE` for details.
