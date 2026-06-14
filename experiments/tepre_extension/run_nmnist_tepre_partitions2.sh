#!/bin/bash -l

#SBATCH --job-name=nmnist_tepre_p2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=memory
#SBATCH --mem-per-cpu=30GB
#SBATCH --account=education-eemcs-msc-dsait
#SBATCH --output=/scratch/tjsorobka/FRML_Reproduction/logs/%x-%j.out
#SBATCH --error=/scratch/tjsorobka/FRML_Reproduction/logs/%x-%j.err

module purge
module load 2025
module load python
module load gettext

cd /scratch/tjsorobka/FRML_Reproduction

source .venv/bin/activate

echo "Python used:"
which python
python --version

echo "Testing imports:"
python -c "import numpy, sklearn, tonic; print('imports ok')"

echo "Starting experiment: n_partitions=2"

python experiments/tepre_extension/run_nmnist_tepre_partition_sweep.py 2