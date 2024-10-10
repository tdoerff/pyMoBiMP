# pyMoBiMP

Author: Tom Doerffel (2024)

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
conda install colorcet fenics-dolfinx fenics-libdolfinx gmsh "h5py>=3.11=mpi*" imageio imageio-ffmpeg ipympl jupyter jupyter pytest pytest-cov python-gmsh pyvista scifem scipy tqdm vtk
```

Activate the ```widgets```extension for a seamless integration into ```VSCode```.
```{bash}
jupyter labextension enable --py widgetsnbextension
````

- finally, install the project packed in developer's mode
```{bash}
pip install -e .
```