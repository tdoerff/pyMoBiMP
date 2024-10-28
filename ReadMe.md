# pyMoBiMP

**Author:** Tom Doerffel (2024)

Special thanks to Manuel Landstorfer for his helpful advice and guidance in the development of this project.

## Project Overview

This code simulates a thermodynamically consistent (half-cell) battery model to compute the intercalation of lithium in active materials such as graphite.

### Directory Structure

- **demos/**: Simple examples demonstrating the basic functionalities.
- **experiments/**: More complex experiments showcasing advanced applications.
- **pyMoBiMP/**: Main code of the project.
- **tests/**: Contains unit tests.

## Installation (conda)

An Anaconda installation is required.
Within the project's base dir, do the following steps:

- create a conda environment
```{bash}
conda create -n fenicsx-env
conda activate fenicsx-env
conda config --add channels conda-forge
conda config --set channel_priority strict
```

If there are any issues with blocked channels, run

```{bash}
conda config --set allow_non_channel_urls True
```
to skip channels that are unavailable.

- install all necessary packages
```{bash}
conda install colorcet fenics-dolfinx fenics-libdolfinx gmsh "h5py>=3.11=mpi*" imageio imageio-ffmpeg ipympl jupyter pytest pytest-cov python-gmsh pyvista scifem scipy tqdm vtk
```

Activate the ```widgets```extension for a seamless integration into ```VSCode```.
```{bash}
jupyter labextension enable --py widgetsnbextension
````

- finally, install the project packed in developer's mode
```{bash}
pip install -e .
```

## Usage

After installation, you can run the provided examples in the demos/ and experiments/ directories to explore the functionalities of the model.
The model usally outputs a collection of files (`*.h5`, `*.xdmf`, and `*_rt.txt`) containing the grid output and processed output of charge and cell voltage.
To visualize the output, there is a script `tools/create_multi_particle_animation.py`that allows visualizing the `XDMF`and `*_rt.txt`output. Use

```
python tools/create_multi_particle_animation.py -h
```
for a detailed instruction.

## Tests

Tests can be run using `pytest`:

````
pytest tests/
````
or with
```
pytest --cov tests/
```
to get a coverage report.

## Contributions

Contributions are welcome! Please open an issue or create a pull request to suggest improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Acknowledgments

This project uses FEniCSx, which is licensed under the GNU Lesser General Public License version 3 (LGPL v3).
Please refer to the FEniCSx [website](https://fenicsproject.org/) for more information.
This project also uses GMSH, which is licensed under the GNU General Public License (GPL).
Please refer to the GMSH [website](http://gmsh.info) for more information.

This project was supported by funding from the German Research Foundation (DFG) through the Excelence Cluster [Math+](http://www.mathplus.de/), project [PaA-2](https://mathplus.de/transfer/partnership-area-paa-2/).
