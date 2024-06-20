#!/bin/bash
#SBATCH -J Uncertain3D
#SBATCH --array=40-44
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --gres=gpu
#SBATCH -o singleRun3d.out

module load anaconda3 nvidia
conda activate ai

DENUMINATOR=16

AVAILABLE_FILTERS=( "ramp" "shepp-logan" "cosine" "hamming" "hann" )

IDX=$SLURM_ARRAY_TASK_ID
#IDX=44

FILTER_IDX=$(bc -l <<< "oldscale=scale; scale=0; ${IDX} % 5; scale=oldscale;")
ANGLE_IDX=$(bc -l <<< "${IDX}/5")
if [[ 1 -eq $(echo "$ANGLE_IDX >= 1.0" | bc -l) ]]
then
    ANGLE_IDX=${ANGLE_IDX%.*}
else
    ANGLE_IDX=0
fi

NUM_ANGLE=$(bc -l <<< "180 * (${ANGLE_IDX}+8)/${DENUMINATOR}")
NUM_ANGLE=${NUM_ANGLE%.*}

echo "python one-shot-3d-uncertainty.py ${NUM_ANGLE} 1 130 ${AVAILABLE_FILTERS[$FILTER_IDX]}"
