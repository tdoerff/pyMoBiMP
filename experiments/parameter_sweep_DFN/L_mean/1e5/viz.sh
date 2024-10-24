#/bin/bash

# adjust accordingly!
PYMOBIMP_BASE_DIR=${HOME}/Documents/Projects/single-particle-cahn-hilliard

python ${PYMOBIMP_BASE_DIR}/tools/create_multi_particle_animation.py output -o anim.mp4 -e experiment --clipped
