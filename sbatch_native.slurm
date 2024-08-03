#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=debug64c512g
#SBATCH -n 128
#SBATCH --ntasks-per-node=64

module load dealii/9.4.0-gcc-9.3.0-openmpi

cd ~/lxl/pfm-dealii
# rm -rf build-native
# mkdir build-native
cd build-native
# cmake ..
# make release
# make
srun -n 16 ./main ../parameters/singleNotchTension.prm

