#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  

#SBATCH -J Cholec80 #Job name

#SBATCH -t 72:00:00 #Maximum runtime of 168 hours

# Enable email notificaitons when job begins and ends
#SBATCH --mail-user=syangcw@connect.ust.hk #Update your email address
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Choose partition (queue) with "gpu"
#SBATCH -p normal

# To use 96 cpu core and 8 gpu devices in a node
#SBATCH -N 1 -n 12 --gres=gpu:2

# output file
#SBATCH --output=Cholec80_%j.out
#SBATCH --error=Cholec80_%j.err
#SBATCH --account=medimgfmod

# Setup runtime environment if necessary
module load gcc slurm
module load cuda11.8

source activate SurgSSL
# Go to the job submission directory and run your application
sh scripts/Natural/train_phase_unified_model_cholec80_video_kd.sh
# sh scripts/triplet/train_phase_unified_model_prostate21_video.sh
