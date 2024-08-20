# Makefile

# Default values for environment variables
ENV_NAME ?= default_env
ENV_MODULE ?= default_env_module
CUDA_DEVICE ?= 0

# Define the commands
TRAIN_CMD=CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 train.py --env_name $(ENV_NAME) --env_module $(ENV_MODULE)
INFER_CMD=CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 infer.py --env_name $(ENV_NAME) --env_module $(ENV_MODULE)
CONVERT_CMD=TF_USE_LEGACY_KERAS=1 CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 convert.py --env_name $(ENV_NAME)

TRAIN_WRAPPER_CMD=CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 train_wrapper.py --env_name $(ENV_NAME) --env_module $(ENV_MODULE)
INFER_WRAPPER_CMD=CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 infer_wrapper.py --env_name $(ENV_NAME) --env_module $(ENV_MODULE)
CONVERT_WRAPPER_CMD=TF_USE_LEGACY_KERAS=1 CUDA_VISIBLE_DEVICES=$(CUDA_DEVICE) python3 convert_wrapper.py --env_name $(ENV_NAME)

# Default target
# Example usage: make all ENV_NAME=my_env CUDA_DEVICE=1
all: train infer convert

train:
	$(TRAIN_CMD)

infer: train
	$(INFER_CMD)

convert: infer
	$(CONVERT_CMD)

# Wrapper target
# Example usage: make all_wrapper ENV_NAME=my_env CUDA_DEVICE=1
all_wrapper: train_wrapper infer_wrapper convert_wrapper

train_wrapper:
	$(TRAIN_WRAPPER_CMD)

infer_wrapper: train_wrapper
	$(INFER_WRAPPER_CMD)

convert_wrapper: infer_wrapper
	$(CONVERT_WRAPPER_CMD)