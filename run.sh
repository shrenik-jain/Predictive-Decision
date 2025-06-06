# !/bin/bash

source /src/code/start.sh
echo "Active Conda Environment: $(conda info | grep 'active environment')"
cd /src/
source /src/code/smarts.sh
cd /src/code/
source source_installations.sh

nvidia-smi

echo "**************************************************************************"
echo "Started Training/Testing"
echo "**************************************************************************"

python train.py --use_exploration --use_interaction
# python test.py --model_path models/lstm_predictor_0.6651.pth --use_interaction --decoder lstm

echo "**************************************************************************"
echo "Completed Training/Testing"
echo "**************************************************************************"