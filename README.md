# MasterThesis

## Overview (Work in progress)

This repository contains the work related to my Master Thesis project focused on applying machine learning techniques to Terahertz Spectroscopy data.

Two main workflows are used in this project and reflected in the project structure:
1. **Notebook-based workflow for initial experimentation**: This workflow is used in the `.ipynb` files at the uppermost layer of the repository, such as 'ExploringAirPulses.ipynb'.
2. **Modular Pipeline for a more structured approach**: After work on a notebook finishes, it is split into steps like preprocessing, feature engineering, etc., and run using the MLFlow framework. MLFlow stores the pipeline along with a snapshot of the project environment with all dependencies and scripts. This framework ensures reproducibility for the experiments and helps with keeping track of everything. The pipeline can be seen under the 'scripts' folder. MLFlow files are found under 'mlruns' and 'mlflow_data' folders.
