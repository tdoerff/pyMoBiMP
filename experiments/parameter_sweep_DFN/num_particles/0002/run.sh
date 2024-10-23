touch running.lock
python experiment.py 2>&1 > output.log
rm running.lock
