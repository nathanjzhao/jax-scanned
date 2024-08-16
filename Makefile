# Makefile

# Default values for environment variables
ENV_NAME ?= default_env
CUDA_DEVICE ?= 0

# Define the commands
TRAIN_CMD=CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 train.py --env_name $(ENV_NAME)
INFER_CMD=CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 infer.py --env_name $(ENV_NAME)
CONVERT_CMD=TF_USE_LEGACY_KERAS=1 CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 convert.py --env_name $(ENV_NAME)

# Default target
# Example usage: make all ENV_NAME=my_env CUDA_DEVICE=1
all: train infer convert

train:
    $(TRAIN_CMD)

infer: train
    $(INFER_CMD)

convert: infer
    $(CONVERT_CMD)