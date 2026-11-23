# NanoShieldTrapColumnDataAnalysis

This repository features the notebooks used for data analysis and post processing for the NanoShield trap column studies.

## Preprint available

### "Trapping more seeing more, benchmarking a new trap column for low-input proteomics"

Dennis Dannecker, Todd Blackburn, Liang Chen, Jarrod J. Sandow, Karl Mechtler, Manuel Matzinger

[https://doi.org/10.64898/2026.11.12.766175](https://doi.org/10.64898/2026.11.12.766175)

## Notes

- Each post processing analysis has its dedicated Jupyter Notebook.
- Each notebook exports `.csv` files with data for external visualization.
- Place notebooks (.ipynb) and python files (.py) in the same directory.

## How to run respective Jupyter Notebooks (recommended)

1. Create a conda environment

   ```bash
   conda env create -f environment.yml
   ```

   The created environment will be called `trap_column_test`.

2. Activate the newly created conda environment

   ```bash
   conda activate trap_column_test
   ```

3. Make the newly created conda environment available as a kernel in Jupyter

   ```bash
   python -m ipykernel install --user --name=trap_column_test
   ```

With this, the environment should be available as a kernel in Jupyter (called "trap_column_test"), and the different notebooks can be run.
