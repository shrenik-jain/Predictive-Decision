#!/bin/bash

# conda install -c conda-forge matplotlib scipy imageio wandb pytorch-lightning xarray netcdf4 dask hydra-core xbatcher boto3 zarr cachey -y
# pip install einops tensordict timm cartopy
# pip install --user torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# # pip install torch torchvision --upgrade
# conda install -c conda-forge imageio ffmpeg -y

pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip uninstall charset_normalizer
pip install "charset_normalizer<3"  
pip install "typing-extensions==4.5.0"
pip install "pathos==0.2.8" "tabulate>=0.8.10" waymo-open-dataset-tf-2-4-0

echo "All dependencies installed in Conda environment: wormhole"