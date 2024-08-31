FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python3-venv build-essential \
    cmake ninja-build git wget

# Upgrade pip
RUN pip3 install --upgrade pip

# Clone the Triton repository
RUN git clone https://github.com/openai/triton.git /workspace/triton

# Set the working directory
WORKDIR /triton_build/triton

# Install build dependencies
RUN pip3 install ninja cmake wheel

# Add the setup.py to the directory if it's not there already
COPY setup.py /triton_build/triton/
COPY main.py /triton_build/triton

RUN pip3 install python-dotenv colorama datasets accelerate
RUN pip3 install liger-kernel pandas 

# Build and install Triton
RUN pip3 install .

