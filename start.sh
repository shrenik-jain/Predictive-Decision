#!/bin/bash

# Start SSH service
/usr/sbin/sshd

# Add Conda to the PATH and initialize Conda
export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh
chmod g+w -R /opt/conda/* 

# Check if Conda environment exists, else create it
conda create -n smarts python=3.7 -y

echo "Conda environment setup complete."

conda activate smarts

apt-get update && apt-get install -y zip unzip