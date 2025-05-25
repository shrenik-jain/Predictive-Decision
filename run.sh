# !/bin/bash

source /src/code/start.sh
echo "Active Conda Environment: $(conda info | grep 'active environment')"
cd /src/
source /src/code/smarts.sh
cd /src/code/
source source_installations.sh

nvidia-smi

echo "**************************************************************************"
echo "Starting training with SMARTS..."
echo "**************************************************************************"

python train.py --use_exploration --use_interaction

echo "**************************************************************************"
echo "Training completed successfully."
echo "**************************************************************************"
ls /src/code

zip /src/results/training_log.zip -r /src/code/training_log/