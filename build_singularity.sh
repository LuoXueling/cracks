cd ~/lxl/pfm-dealii
singularity instance start ../dealii-9.4.0.sif dealii1
rm -rf build-singularity
mkdir build-singularity
cd build-singularity
singularity exec instance://dealii1 cmake ..
singularity exec instance://dealii1 make release
singularity exec instance://dealii1 make
singularity instance stop dealii1

