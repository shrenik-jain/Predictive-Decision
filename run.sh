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
python test.py --model_path models/transformer_1500epochs_predictor_0.6500.pth --use_interaction --decoder lstm

echo "**************************************************************************"
echo "Completed Training"
echo "**************************************************************************"

# zip /src/results/trans_training_log.zip -r /src/code/training_log/
zip /src/results/tf_1500_test_log.zip -r /src/code/test_log/