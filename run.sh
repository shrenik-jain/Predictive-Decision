# !/bin/bash

source /src/code/start.sh
echo "Active Conda Environment: $(conda info | grep 'active environment')"
cd /src/
source /src/code/smarts.sh
cd /src/code/
source source_installations.sh

nvidia-smi

echo "**************************************************************************"
echo "Started Training"
echo "**************************************************************************"

# python train.py --use_exploration --use_interaction
python test.py --model_path models/transformer_predictor_0.8580.pth --use_interaction --decoder transformer

echo "**************************************************************************"
echo "Completed Training"
echo "**************************************************************************"

# zip /src/results/training_log.zip -r /src/code/training_log/
zip /src/results/trans_test_log.zip -r /src/code/test_log/