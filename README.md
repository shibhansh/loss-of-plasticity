# Loss of Plasticity in Deep Continual Learning

This repository contains the implementation of three continual supervised learning problems.
In our forthcoming paper _Loss Plasticity in Deep Continual Learning_,
we show the loss of plasticity in deep learning in these problems.

A talk about this work can be found [here](https://www.youtube.com/watch?v=p_zknyfV9fY), 
and the [paper](https://arxiv.org/abs/2306.13812) is available on arxiv.

# Installation

```sh
mkdir ~/envs
virtualenv --no-download --python=/usr/bin/python3.8 ~/envs/lop
source ~/envs/lop/bin/activate
pip3 install --no-index --upgrade pip
git clone https://github.com/shibhansh/loss-of-plasticity.git
cd loss-of-plasticity
pip3 install -r requirements.txt
pip3 install -e .
```

Add these lines in your ~/.zshrc or ~/.bashrc
```sh
source ~/envs/lop/bin/activate
```