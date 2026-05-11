#!/bin/bash
#SBATCH --job-name=brain_uq_v2
#SBATCH --partition=gpu-a6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --account=miazhang
#SBATCH --output=/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/uncertainty_brain_sde_v2/loginfo/train_bridgeuq_%j.log
#SBATCH --error=/sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/uncertainty_brain_sde_v2/loginfo/train_bridgeuq_%j.log

# Override at submission time, e.g.:
#   BACKBONE=ltma BATCH_SIZE=4 DATASET=full sbatch slurm_train_bridgeuq.sh
BACKBONE=${BACKBONE:-voxelmorph}
BATCH_SIZE=${BATCH_SIZE:-57}
DATASET=${DATASET:-mni88}

source activate voxelmorph
module load gcc

cd /sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2

mkdir -p /sfs/weka/scratch/swd9tc/Uncertanity_quantification/LightingTemplate_2/uncertainty_brain_sde_v2/loginfo

echo "=== Brain BridgeUQ v2 Training ==="
echo "Backbone:    $BACKBONE"
echo "Batch size:  $BATCH_SIZE"
echo "Dataset:     $DATASET"
echo "Loss:        mse"
echo "Start:       $(date)"
echo "Node:        $(hostname)"
nvidia-smi

python -m uncertainty_brain_sde_v2.main \
    --backbone "$BACKBONE" \
    --batch_size "$BATCH_SIZE" \
    --dataset "$DATASET" \
    --loss mse

echo "=== Done: $(date) ==="
