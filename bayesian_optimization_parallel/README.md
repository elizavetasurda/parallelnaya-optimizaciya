# Bayesian Optimization Parallel

This project implements parallel Bayesian optimization using Gaussian Processes.

## Structure

- `main.py`: Main entry point for running the optimization.
- `requirements.txt`: Python dependencies.
- `src/`: Source code modules.
  - `gp_model.py`: Gaussian Process model.
  - `acquisition.py`: Acquisition functions (EI, UCB).
  - `parallel_methods.py`: Methods for parallel evaluation.
  - `test_functions.py`: Benchmark test functions.
  - `utils.py`: Utility functions for plotting and saving.
- `results/`: Directory for storing optimization results.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the main script:

```bash
python main.py
```

This will optimize the Rosenbrock function and save results to `results/`.

## License

MIT