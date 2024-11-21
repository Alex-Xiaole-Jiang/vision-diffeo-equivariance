#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=Unet
#SBATCH --mail-type=END
#SBATCH --mail-user=xj2173@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err


module purge


singularity exec --nv \
  --overlay /vast/xj2173/diffeo-env.ext3:ro \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-val.sqf:ro      \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-test.sqf:ro     \
  --overlay /vast/work/public/ml-datasets/imagenet/imagenet-train.sqf:ro    \
  /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -c "source /ext3/env.sh; python /home/xj2173/vast/diffeo/U-net-reconstruction.py"