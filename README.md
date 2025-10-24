# Data-Driven Stabilization Using Prior Knowledge on Stabilizability and Controllability

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

This repository provides a **MATLAB implementation** of data-driven control for stabilizable systems, demonstrated on a **three-tank process**.
It supports:

- Experiment design with random input trajectories
- Data-driven controller synthesis using LMI-based methods
- Monte Carlo analysis for dataset informativity
- Closed-loop simulations for visualization

---

## Authors

- **Amir Shakouri** – University of Groningen

- **Henk J. van Waarde** – University of Groningen

- **Tren M.J.T. Baltussen** – Eindhoven University of Technology

- **W.P.M.H. (Maurice) Heemels** – Eindhoven University of Technology

  *Contact:* t.m.j.t.baltussen@tue.nl

---

## Requirements

- MATLAB R2020b or later
- [YALMIP](https://yalmip.github.io/) (for LMI optimization)
- [SeDuMi](https://sedumi.ie.lehigh.edu/) (solver for LMIs)

---

## Python Reimplementation

The `python/` directory contains a feature-complete translation of the MATLAB
toolbox. It reproduces the system setup, informativity checks and controller
synthesis using NumPy, SciPy, CVXPY and Matplotlib.

### Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python/requirements.txt
python python/numerical_example.py
python python/monte_carlo_analysis.py
```

The scripts mirror `Numerical_Example.m` and `Monte_Carlo_Analysis.m`, leveraging
the reusable functions in `python/data_driven_stabilization/`.

---