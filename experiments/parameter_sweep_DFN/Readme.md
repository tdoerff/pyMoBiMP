This directory contains a number of experiments to test the battery model against different parameter changes.

The subdirectory structure is aligned with the experiments:

- `default/`: experiment with default parameters
- `c-rate/`: probes the charging rate
- `diffusion/`: probes the diffusion coefficient
- `gamma/`: probes the phase separation parameter
- `L_mean`: probes the mean affinity parameter
- `num_particles`: probes the number of particles

Every directory (except `default/`) contains subdirectories that correspond to the parameter value which again contain the corresponding experiment.
Each experiment directory contains three files:

- `experiment.py`: the experiment script (includes all the configuration)
- `run.sh`: a vanilla shell script to run the simulation and create `run.lock`that indicates a running simulation
- `viz.sh`: a shell script calling the visualization script with the proper arguments
