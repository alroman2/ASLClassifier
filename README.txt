1. Install docker before proceeding
2. Follow bellow instructions for your system
# ==================== WINDOWS/Linux INSTRUCTIONS ====================
# NOTE: Make sure you know what cuda version installed on your gpu and change the version of pytorch in the Dockerfile accordingly
# if your base system does not have cude see https://developer.nvidia.com/cuda-toolkit

# TO RUN TRAINING: docker-compose up train
# TO RUN INFERENCE: docker-compose up inference

# If your system does not have an NVIDIA GPU, please use the commands label with "no-gpu", using the non-label commands will cause errors on systems without a GPU
  
# TO RUN TRAINING WITHOUT GPU: docker-compose up train-no-gpu
# TO RUN INFERENCE WITHOUT GPU: docker-compose up inference-no-gpu

# ==================== MACBOOK INSTRUCTIONS ====================
# If you are using a macbook please use the commands label with "Macbook", using the non-label commands will cause errors on macbooks as the target is for windows
# TO RUN TRAINING ON MACBOOK: docker-compose up train-mac
# TO RUN INFERENCE ON MACBOOK: docker-compose up inference-mac

3. For training all files output to the dir "outputs"
4. For inference it takes in data from "test_data" and outputs to "Inference_out"

