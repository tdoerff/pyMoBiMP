#!/bin/bash

# run that script with `source start_xvfb.sh` to set env variables and start the xvfb
# server necessary to run the PyVista visualization script on a compute node without
# running X server.

set -x
export DISPLAY=:99.0
export MESA_GL_VERSION_OVERRIDE=3.2
export PYVISTA_OFF_SCREEN=true
export PYVISTA_USE_IPYVTK=true
which Xvfb
Xvfb :99 -screen 0 1920x1080x24 > /dev/null 2>&1 &
sleep 3
set +x
exec "$@"
