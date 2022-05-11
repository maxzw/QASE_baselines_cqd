#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu_shared
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.j.zwager@student.vu.nl
#SBATCH --output=job_logs/output_%A.out
#SBATCH --error=job_logs/errors_%A.err

module purge all
module load 2021
module load Anaconda3/2021.05

# define and create a unique scratch directory
SCRATCH_DIRECTORY=/global/work/${USER}/kelp/${SLURM_JOBID}
mkdir -p ${SCRATCH_DIRECTORY}
cd ${SCRATCH_DIRECTORY}

# Activate Anaconda work environment for OpenDrift
source /home/${USER}/.bashrc
source activate thesis

# Run your code
srun python main.py --do_valid --do_test --data_path data/FB15k-q2b -n 1 -b 1000 -d 1000 \
--cpu_num 0 --geo cqd --tasks 1p --print_on_screen --test_batch_size 1 --cqd discrete ${@:1}

# --data_path data/FB15k-237-q2b
# --tasks 1p
# --checkpoint_path /logs/AIFB-gqs/1p/cqd/g-cqd/2022.05.11-09:47:42/