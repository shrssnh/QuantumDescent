#!/bin/bash
#SBATCH --output=out.slurm
#SBATCH --error=err.slurm
#SBATCH --ntasks=20
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yashas.b@ada.iiit.ac.in
#SBATCH --mem=105G
#SBATCH -A plafnet2
#SBATCH -p plafnet2
#SBATCH -w gnode114

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate dft

echo "starting..."

# 1. Define scratch directory
SCRATCH_DIR=/scratch/$USER/$SLURM_JOB_ID
echo "scratch directory:"
echo $SCRATCH_DIR

# 2. Create the scratch directory
mkdir -p $SCRATCH_DIR

echo "directory made"

# 3. Copy all files from submission directory to scratch
cp -r ./* $SCRATCH_DIR

echo "contents copied"

# 4. Go to scratch directory
cd $SCRATCH_DIR

./ran.sh
