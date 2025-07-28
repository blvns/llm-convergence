#!/bin/bash
#SBATCH --job-name=accomm-prompt
#SBATCH --partition=p_csunivie_gres
#SBATCH --account=datamining
#SBATCH --nodes=1
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH --time=12:00:00

### Run your job
srun --label python /srv/home/users/blevinst24cs/llm-accommodation/main_LLMs.py "${@}"