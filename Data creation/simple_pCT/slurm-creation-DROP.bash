#!/bin/bash
#SBATCH -J Uncertain3D
#SBATCH --array=0-8
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
##SBATCH --gres=gpu
##SBATCH --time=3:00:00
#SBATCH -o slurm-creation-DROP.out

module load anaconda3
conda activate ai

DENUMINATOR=16

IDX=$SLURM_ARRAY_TASK_ID
# IDX=8

ANGLE_IDX=$IDX

NUM_ANGLE=$(bc -l <<< "180 * (${ANGLE_IDX}+8)/${DENUMINATOR}")
NUM_ANGLE=${NUM_ANGLE%.*}

python one-shot-drop.py --num_angle=${NUM_ANGLE} --num_offset=31 --num_spotx=130 --num_samples=50 --mlp_height=130 --mlp_width=130 --lamb=0.5 --num_iterations=2000
