# Delete the old timing file.
rm -f profiling/dfn_scaling.csv


for n_procs in 1 2 4 8 16 32
do
    for it in {1..4}
    do
    mpirun -np $num_procs python profiling/dfn_scaling_runner.py
    done
done
