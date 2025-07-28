#!/bin/bash
#SBATCH --job-name=accomm-stylo
#SBATCH --partition=p_csunivie_gres
#SBATCH --account=datamining
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --time=1:00:00

### Run your job
srun --label python /srv/home/users/blevinst24cs/llm-accommodation/postprocess_stylometrics.py "${@}"