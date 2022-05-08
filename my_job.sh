#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --partition=gpu_shared
#SBATCH --time=50:00:00
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
srun python main.py --cuda --do_train --do_valid --do_test \
  --data_path data/AIFB-gqs -n 128 -b 512 -d 800 -g 24 \
  -lr 0.0001 --max_steps 450001 --cpu_num 1 --geo vec --valid_steps 15000 \
  --tasks "1p.2p.3p.2i.3i.ip.pi" --checkpoint_path "logs/AIFB-gqs/1p.2p.3p.2i.3i.ip.pi/vec/g-24.0/2022.05.04-22:36:23/"
