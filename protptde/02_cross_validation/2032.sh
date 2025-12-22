#!/usr/bin/env bash

set -Eeuo pipefail

#######################################################################
# conda environment
#######################################################################

__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
conda activate Prot_PTDE

script_name=$(basename "$0")
gpu=${script_name: -4:1}

for i in 16 17; do
    CUDA_VISIBLE_DEVICES="${gpu}" python train.py --random_seed "${i}"
done
