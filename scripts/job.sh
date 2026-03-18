#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --account=aip-medilab
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=
#SBATCH --mem=20G
#SBATCH --nodes=1
#SBATCH --time=8:00:00
#SBATCH --output=/home/harmanan/projects/aip-medilab/harmanan/breast_us/logs/%x_%j.out
#SBATCH --error=/home/harmanan/projects/aip-medilab/harmanan/breast_us/logs/%x_%j.err

JOBPATH=/home/harmanan/projects/aip-medilab/harmanan/breast_us/scripts

bash ${JOBPATH}/run_job.sh "$@"