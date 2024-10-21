set -x

conda activate fenicsx-env

touch running.lock
python CH_4_DFN_grid.py 2>&1 > output.log
rm running.lock
