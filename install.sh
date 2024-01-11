mkdir ~/envs
virtualenv --no-download ~/envs/lop
source ~/envs/lop/bin/activate
pip3 install --no-index --upgrade pip
pip3 install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
pip3 install -e .