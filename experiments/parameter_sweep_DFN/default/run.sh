touch running.lock
~/.conda/envs/fenicsx-env/bin/python CH_4_DFN_grid.py >&1 > output.log
rm running.lock
