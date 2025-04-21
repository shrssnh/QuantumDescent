#!/bin/bash
#SBATCH --job-name=DEEPH-Train
#SBATCH --output=out.slurm
#SBATCH --error=err.slurm
#SBATCH --ntasks=20
#SBATCH --gres=gpu:2
#SBATCH --time=96:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=yashas.b@research.iiit.ac.in
#SBATCH --mem=105G
#SBATCH -w gnode118
#SBATCH -p plafnet2
#SBATCH -A plafnet2


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

# 5. Run your calculation
# Example: run a Python script
deeph-train --config ./default.ini

# 6. Copy results back to submission dir
# cp -r $SCRATCH_DIR/* $SLURM_SUBMIT_DIR/

# 7. Optional: clean up scratch (only if needed)
# rm -rf $SCRATCH_DIR