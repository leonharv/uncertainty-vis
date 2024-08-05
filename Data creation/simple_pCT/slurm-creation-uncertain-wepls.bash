#!/bin/bash
#SBATCH -J Uncertain3D
#SBATCH --array=0-44
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
##SBATCH --gres=gpu
##SBATCH --time=3:00:00
#SBATCH -o slurm-creation-uncertain-wepls.out

module load anaconda3 nvidia
conda activate ai

DENUMINATOR=16

AVAILABLE_FILTERS=( "ramp" "shepp-logan" "cosine" "hamming" "hann" )

IDX=$SLURM_ARRAY_TASK_ID
# IDX=0

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

python one-shot-creation-uncertain-wepls.py --num_angle=${NUM_ANGLE} --num_offset=31 --num_spotx=130 --filter_name=${AVAILABLE_FILTERS[$FILTER_IDX]} --mlp_height=130 --mlp_width=130
