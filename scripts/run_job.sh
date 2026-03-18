#!/bin/bash

# Default values
YAML="cbm"
GROUP="DefaultGroup"
FOLD=0
SEED=0
TIME="24:00:00"
MEM="32G"
GPU="l40s"
NGPU=1

# Parse args passed to sbatch
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yaml)
            YAML="$2"
            shift 2
            ;;
        -g|--group)
            GROUP="$2"
            shift 2
            ;;
        -t|--time)
            TIME="$2"
            shift 2
            ;;
        -m|--mem)
            MEM="$2"
            shift 2
            ;;
        -x|--gpu)
            GPU="$2"
            shift 2
            ;;
        -n|--ngpu)
            NGPU="$2"
            shift 2
            ;;
        -f|--fold)
            FOLD="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done


JOB_NAME="${YAML//\//_}"
TMP_SCRIPT=$(mktemp)

cat <<EOF > $TMP_SCRIPT
#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --account=aip-medilab
#SBATCH --gres=gpu:${GPU}:${NGPU}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --mem=${MEM}
#SBATCH --nodes=1
#SBATCH --time=${TIME}
#SBATCH --exclude=kn072,kn064
#SBATCH --output=/home/harmanan/projects/aip-medilab/harmanan/breast_us/logs/%x_%j.out
#SBATCH --error=/home/harmanan/projects/aip-medilab/harmanan/breast_us/logs/%x_%j.err

JOBPATH=/home/harmanan/projects/aip-medilab/harmanan/breast_us/scripts

mkdir -p /home/harmanan/projects/aip-medilab/harmanan/breast_us/logs

# Run python with injected args
python main.py -y $YAML -o seed=$SEED wandb.group=$GROUP data.fold=$FOLD
EOF

# Submit the job
sbatch $TMP_SCRIPT