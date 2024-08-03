ssh sy
srun -p debug64c512g -N 1 -n 1 --cpus-per-task=64 --pty /bin/bash
cd lxl/pfm-dealii
singularity instance start ../dealii-9.4.0.sif dealii1
singularity exec instance://dealii1 ./compile_and_run.sh -n 16 -r release -f test.prm


