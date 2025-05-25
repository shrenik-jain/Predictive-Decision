ls
git clone https://github.com/huawei-noah/SMARTS.git
echo "SMARTS repository cloned."
ls
cd SMARTS
git checkout comp-1

# Install the system requirements.
bash utils/setup/install_deps.sh

# Install smarts with comp-1 branch.
pip install "smarts[camera-obs] @ git+https://github.com/huawei-noah/SMARTS.git@comp-1"

echo "SMARTS installed successfully."
echo "**************************************************************************"