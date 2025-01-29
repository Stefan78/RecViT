# Installation of conda environment
1. Create conda environment using python 3.8
   - _command_: conda create --name recvit python=3.8
2. Activate the created environment
   - _command_: conda activate recvit
3. Install pytorch, torchvision, torchaudio and pytorch-cuda from the official pytorch website according to your device specifications
   - _versions used during testing_: conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
4. Install UMAP, wandb and tqdm packages via conda-forge
   - _command_: conda install conda-forge::umap-learn
   - _command_: conda install conda-forge::wandb
   - _command_: conda install conda-forge::tqdm
5. Install matplotlib, timm and ART packages via pip
   - _command_: pip install matplotlib==3.7.0
   - _command_: pip install timm==0.9.0
   - _command_: pip install adversarial-robustness-toolbox==1.16.0

# Usage
### config
- Contains paths for data loading/saving. Please see readme.txt in this directory.
### data
- Default folders to load/save data such as networks, adversarial examples and datasets.
### pckgs
- Implementation of various techniques for training, evaluation, plotting and more.
### scripts \ experiments
- scripts for running evaluations
### scripts \ generate_adversarials
- generating and cross-validating adversarial examples
### scripts \ train_CNNs
- training CNNs that were used for generating the adversarial examples
### scripts \ train_nets
- training various types of RecViT