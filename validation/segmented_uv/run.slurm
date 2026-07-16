#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --job-name=radmc
#SBATCH --output=zz_radmc_%A
#SBATCH --time=0-48:00:00
#SBATCH --mem=64G

echo "HOSTNAME = $HOSTNAME"
echo "HOSTTYPE = $HOSTTYPE"
echo Time is $(date)
echo Directory is $(pwd)

module purge all

module load python-scientific/3.13.1-foss-2025a; . /fred/oz015/jcalcino/python_env/bin/activate
python run.py
