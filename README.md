# Neural-Game-Engine
Code to reproduce Neural Game Engine experiments and pre-trained models


# Setup

Create a conda environment using the following command:
`conda create --name 'NGE' python=3.7`

Activate the environment
`conda activate NGE`

Install pytorch (and cuda)
`conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`

Install python requirements

## GVGAI

Clone the GVGAI library from `https://github.com/Bam4d/GVGAI_GYM` and install it into your python environment.

Clone:
`git clone https://github.com/Bam4d/GVGAI_GYM.git`

Navigate to the `GVGAI_GYM/python/gvgai` directory (this is where the GVGAI python library is located).

Install it into your python environment by using:
`pip install -e .`

# Running

## (Optional) Start tensorflow 
To start tensorflow and track the training run the following command in the directory where you have cloned the Neural Game Engine
`./tb.sh`

## Train a game

For example training GVGAI sokoban for 5000 epochs:
`python train.py -e 5000 -G sokoban`

Once the game has trained it is assigned a random ID and then is stored in `gym/pre-trained/[ID]`

## Play with a pre-trained model

For example playing level 3 of the pre-trained model with Id [12345]
`python play.py --level 3 --model '12345'`
