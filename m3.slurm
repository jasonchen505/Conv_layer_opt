#!/bin/bash
#SBATCH --job-name="Milestone3"
#SBATCH --output="Milestone3.out"
#SBATCH --error="Milestone3.err"
#SBATCH --partition=gpuA40x4
#SBATCH --mem=12G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint="projects"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --account=bche-delta-gpu
#SBATCH -t 00:25:00

module reset
module load cuda

echo -e "job $SLURM_JOBID is starting on `hostname`\n\n"

# srun ./m3 10000 > m3.out
{ time srun ./m3 100 > m3_op2_100.out; } 2> time_m3_op2_100.out
{ time srun ./m3 1000 > m3_op2_1000.out; } 2> time_m3_op2_1000.out
{ time srun ./m3 10000 > m3_op2_10000.out; } 2> time_m3_op2_10000.out

# Generate nsys profile information, output will be in profile.out
# srun nsys profile --stats=true ./m3 > profile.out

# Generate analysis_file.ncu-rep with a detailed profiling result
# ncu --set full -f -o analysis_file ./m3 > m3.out