# !/bin/bash

source /src/code/start.sh
echo "Active Conda Environment: $(conda info | grep 'active environment')"
cd /src/
source /src/code/smarts.sh
cd /src/code/
source source_installations.sh

python train.py --use_exploration --use_interaction

zip /src/results/training_logs.zip -r training_logs/