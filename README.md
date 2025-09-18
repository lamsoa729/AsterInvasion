
![Simulation snapshot](assets/unstable_side_snap_t660_wbg.png)
# AsterInvasion
**AsterInvasion** is a Python package for simulating the interaction and invasion of branching microtubule networks in 2D and 3D developed for the paper [Robust cytoplasmic partitioning by solving an intrinsic cytoskeletal instability](https://www.biorxiv.org/content/10.1101/2024.03.12.584684v1) by Rinaldin _et al_.

## Features
- GPU acceleration using Numba CUDA for computationally intensive routines (optional — CPU-only runs are supported)
- Inhibitory interactions between microtubules
- Kinetic Monte Carlo simulation of microtubule dynamics and nucleator binding
- Nucleator binding and filament nucleation
- 2D and 3D systems
- Optional filament–filament and filament–boundary interactions

## Installation
We use [Conda](https://docs.conda.io/en/latest/) to manage the Python environment and dependencies. Make sure you have conda installed (either through [Anaconda](https://www.anaconda.com/products/distribution), [Miniconda](https://docs.conda.io/en/latest/miniconda.html), or [Miniforge](https://github.com/conda-forge/miniforge)).  

Once you have conda set up, follow these steps to install the package:

1. Clone the repository
    ```bash
    git clone https://github.com/lamsoa729/aster_invasion.git
    cd aster_invasion
    ```

2. Create the conda environment from the provided `yml` files:
    ```bash
    # For non-GPU systems (e.g. Mac)
    conda env create -f aster_invasion_env.yml
    # For NVIDIA GPU systems with CUDA
    conda env create -f aster_invasion_env_gpu.yml
    ```
    then activate the environment:
    ```bash
    conda activate aster_invasion
    ```

3. The package will be installed automatically when the environment is created. You can now run the program from the command line

    ```bash
    aster_invasion <option> <config_file.yaml>
    ```

   To see available options, run
    ```bash
    aster_invasion --help
    ```

### Troubleshooting
If packages are missing, try updating conda and cleaning the cache by running

```bash
conda update conda
conda clean --all
conda env update -f <environment_file.yml>
```

For platform-specific issues, check package availability

```bash
conda search -c conda-forge package-name
```

Feel free to send an email or open an issue on GitHub if any problems are encountered.

## Testing

Once installed, run:

```bash
pytest
```

This runs a series of tests to ensure proper functionality.

* Tests do **not** require a GPU. They are designed to run on any system, including laptops and Macs.
* If you have a GPU available, additional GPU routines will be tested automatically.

Failed tests could indicate an installation issue or a bug. If you encounter failed tests, please open an issue on GitHub.

## Quick Start

An example script to run a fast simulation is provided in the `examples/single_centrosome_no_interaction_2d` directory. To run the example:

```bash
cd examples/single_centrosome_no_interaction_2d
aster_invasion single_cent.yaml
```

This will run a 2D simulation (CPU-only) of a single centrosome nucleating microtubules with no MT–MT inhibitory effects. The simulation runs in roughly real time (\~1 minute) and outputs the file `single_cent.h5`

This file is in **HDF5 format**, which can be read with Python libraries such as [`h5py`](https://www.h5py.org/), [`pandas`](https://pandas.pydata.org/), or with tools like [HDFView](https://www.hdfgroup.org/downloads/hdfview/).

A Jupyter notebook, `notebooks/01-single_centrosome_analysis.ipynb`, demonstrates how to read in and analyze the output data. To run it:

```bash
cd notebooks
jupyter notebook 01-single_centrosome_analysis.ipynb
```

The notebook will guide you through loading the data and performing basic analyses of the microtubule density profiles.

## Visualization

3D simulation images in the paper were generated with [ParaView](https://www.paraview.org/). ParaView requires `.vtk` files, so the package includes functionality to convert `.h5` output into `.vtk` format:

```bash
aster_invasion -h2v <output_file.h5>
```

This creates a `vtk/` directory containing `.vtk` files for each time point. These can be opened directly in ParaView. Python scripts for generating images and movies from `.vtk` files are provided in `aster_invasion/paraview` and must be run from within ParaView’s Python environment.

![ParaView screenshot](assets/snapshot.png)

## Paper Figures

Jupyter notebooks for reproducing figures from Rinaldin *et al* are provided in the `notebooks/` directory. These assume the relevant simulation outputs are available as `.h5` files.

Single parameter files for representative simulations are included under:

* `examples/unstable_3d_asters/double_cent_inhib.yaml`
* `examples/stable_3d_asters/double_cent_inhib.yaml`

To generate multiple realizations, vary the `rng_seed` parameter in the YAML file and run in separate directories to avoid overwriting data.

Running 3D simulations is computationally expensive. Analyzed data used to generate the paper figures is included in `notebooks/` so you can reproduce figures without rerunning simulations. To perform new runs, we recommend a high-performance computing cluster with GPU nodes.

## Future Features

* [ ] Removal of PyTorch dependency
* [ ] Better restart functionality
* [ ] State of system stored as a compound NumPy data type
* [ ] Periodic boundary conditions
* [ ] Spherical boundary conditions
* [ ] Mechanics and forces
* [ ] Hydrodynamic field calculations

## Citing

If you use this package in your work, please cite:

Rinaldin *et al*. **Robust cytoplasmic partitioning by solving an intrinsic cytoskeletal instability.** bioRxiv (2024). [https://doi.org/10.1101/2024.03.12.584684](https://doi.org/10.1101/2024.03.12.584684)

## Contributing

Contributions are welcome! If you would like to improve the code, add new features, or report issues:

* Please open an issue to discuss potential changes.
* For code contributions, fork the repository and submit a pull request.
* Ensure that all tests pass locally by running `pytest` before submitting.

We encourage contributions from both experimental and computational researchers who want to extend the functionality of **AsterInvasion**.

## License

Apache Software License 2.0

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [`pyOpenSci/cookiecutter-pyopensci`](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template, based off [`audreyr/cookiecutter-pypackage`](https://github.com/audreyr/cookiecutter-pypackage).

