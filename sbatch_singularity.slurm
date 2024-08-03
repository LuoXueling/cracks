#!/bin/bash

#SBATCH --job-name=test
#SBATCH --partition=debug64c512g
#SBATCH -n 128
#SBATCH --ntasks-per-node=64

cd ~/lxl/pfm-dealii
# singularity instance start ../dealii-9.4.0.sif dealii1
# rm -rf build-singularity
# mkdir build-singularity
cd build-singularity
# singularity exec instance://dealii1 cmake ..
# singularity exec instance://dealii1 make release
# singularity exec instance://dealii1 make
# singularity instance stop dealii1
srun -n 16 --mpi=pmix singularity exec ../../dealii-9.4.0.sif ./main ../parameters/singleNotchTension.prm

