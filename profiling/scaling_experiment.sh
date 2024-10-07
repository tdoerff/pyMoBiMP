# Delete the old timing file.
rm -f profiling/dfn_scaling.csv

~/.conda/envs/fenicsx-env/bin/mpirun -np 1 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 1 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 1 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 1 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py

~/.conda/envs/fenicsx-env/bin/mpirun -np 2 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 2 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 2 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 2 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py

~/.conda/envs/fenicsx-env/bin/mpirun -np 4 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 4 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 4 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 4 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py

~/.conda/envs/fenicsx-env/bin/mpirun -np 8 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 8 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 8 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 8 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py

~/.conda/envs/fenicsx-env/bin/mpirun -np 16 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 16 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 16 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 16 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py

~/.conda/envs/fenicsx-env/bin/mpirun -np 32 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 32 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 32 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 32 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py

~/.conda/envs/fenicsx-env/bin/mpirun -np 32 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 32 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 32 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
~/.conda/envs/fenicsx-env/bin/mpirun -np 32 ~/.conda/envs/fenicsx-env/bin/python profiling/dfn_scaling_runner.py
