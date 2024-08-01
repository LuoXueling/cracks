ssh sy
srun -p debug64c512g -N 1 -n 1 --cpus-per-task=64 --pty /bin/bash
cd lxl
singularity instance start ./dealii.sif dealii1
singularity shell instance://dealii1
cd pfm-dealii
./compile_and_run.sh -n 16 -r release

