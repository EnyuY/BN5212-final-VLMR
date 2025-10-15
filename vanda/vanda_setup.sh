#!/bin/bash

# directory
WORKDIR="/scratch/$USER/my_project/VLMR_dev"
cd $WORKDIR || exit 1

# Singularity
IMAGE="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif"
ENV_DIR="/scratch/$USER/envs/llm-rag"
JDK_DIR="/scratch/$USER/jdk8u402-b06"

# CACHE
CACHE_DIR="$WORKDIR/hf/hf_cache"
mkdir -p "$CACHE_DIR"
export HF_HOME="$CACHE_DIR" TRANSFORMERS_CACHE="$CACHE_DIR" HF_DATASETS_CACHE="$CACHE_DIR"

# Singularity
module load singularity

