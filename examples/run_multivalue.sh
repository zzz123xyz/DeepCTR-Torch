#!/bin/bash
# Created by the University of Melbourne job script generator for SLURM
# Sat Mar 06 2021 14:01:19 GMT+1100 (Australian Eastern Daylight Time)

# Partition for the job:
#SBATCH --partition=deeplearn

# Partition for the qos:
#SBATCH --qos=gpgpudeeplearn

# Multithreaded (SMP) job: must run on one node
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="run_multivalue_movielens_1m_8_1_1"

# The project ID which this job should run under:
#SBATCH --account="punim1395"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# Number of GPUs requested per node:
#SBATCH --gres=gpu:v100:1
# The amount of memory in megabytes per process in the job:
#SBATCH --mem=16000

# Use this email address:
#SBATCH --mail-user=liangchen.liu@unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=1-0:0:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
   echo "You need to submit your job to the queuing system with sbatch"
   exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load gcc
#module load tensorflow
# The job command(s):
source ~/.bashrc
module list
conda activate deepctr
#stdbuf -i0 -o0 -e0 python run_multivalue_movielens_1m_8_1_1.py
#stdbuf -i0 -o0 -e0 python run_multivalue_movielens_1m_8_1_1.py
#stdbuf -i0 -o0 -e0 python run_multivalue_amazon_movies_TV_8_1_1_binary_balance.py
#python -u run_multivalue_amazon_movies_TV_8_1_1_binary_balance.py
#python -u run_multivalue_movielens_1m_8_1_1_two_features_correct_lbecoder.py
#python -u run_multivalue_movielens_1m_8_1_1_correct_lbeencoder.py
python -u run_multivalue_movielens_1m_binary_balance_correct_lbeencoder.py
