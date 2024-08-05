python ../../../tools/create_multi_particle_animation.py simulation_output/output -r 0.2 -o anim.mp4 --clipped -e experiment.py

ffmpeg -y -i anim.mp4 -filter_complex "fps=10,scale=500:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=64[p];[s1][p]paletteuse=dither=bayer" anim.gif
