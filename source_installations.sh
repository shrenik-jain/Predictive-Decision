#!/bin/bash

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip uninstall charset_normalizer
pip install "charset_normalizer<3"  
pip install "typing-extensions==4.5.0"
pip install "pathos==0.2.8" "tabulate>=0.8.10" waymo-open-dataset-tf-2-4-0

echo "All dependencies installed in Conda environment: smarts"
echo "**************************************************************************"