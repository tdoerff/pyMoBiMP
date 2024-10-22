# Delete the old timing file.
rm -f profiling/dfn_scaling.csv

# Activate the conda environment
conda activate fenicsx-env

# Display the paths for mpirun and python
which mpirun
which python

# Define the number of processors
for n_procs in 1 2 4 8 16 32
do
    for it in {1..4}
    do
        mpirun -np ${n_procs} python profiling/dfn_scaling_runner.py
    done
done