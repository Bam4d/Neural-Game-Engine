# Neural-Game-Engine
Code to reproduce Neural Game Engine experiments and pre-trained models


# Setup

Firstly clone the Neural Game Engine repository:

```
git clone git@github.com:Bam4d/Neural-Game-Engine.git
```

Create a conda environment using the following command:
```
conda create --name 'NGE' python=3.7
```

Activate the environment
```
conda activate NGE
```

Install pytorch (and cuda)
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Install python requirements

```
cd ./Neural-Game-Engine
pip install -r requirements.txt
```

## GVGAI

GVGAI is required to reproduce the training of any of the experiments.

Clone the GVGAI library from `https://github.com/Bam4d/GVGAI_GYM` and install it into your python environment.

Clone:
```
git clone git@github.com:Bam4d/GVGAI_GYM.git
```

Navigate to the `GVGAI_GYM/python/gvgai` directory (this is where the GVGAI python library is located).

Install it into your python environment by using:
```
pip install -e .
```

# Running

## (Optional) Start tensorflow 
To start tensorflow and track the training run the following command in the directory where you have cloned the Neural Game Engine
```
./tb.sh
```

## Train

For example training GVGAI sokoban for 5000 epochs:
```
python train.py -e 5000 -G sokoban
```

Once the game has trained it is assigned a random ID and then is stored in `gym/pre-trained/[ID]`

## Play

To test out the pre-trained environments, you can play them using your keyboard. 
W,A,S,D keys and space are mapped to up, left, down, right and use respectively.

For example playing level 3 of the pre-trained model with Id `308e9d7c-371f-495d-9f53-95170220e5b1`
```
python play.py --level 3 --id "308e9d7c-371f-495d-9f53-95170220e5b1"`
```

## Statistical Forward Planning

An example is provided to show the rolling horizon evolutional algorithm using the neural game engine in inference mode 
for rollouts.

`python rhea.py -R --level "gvgai-sokoban-lvl4-v0" --id "308e9d7c-371f-495d-9f53-95170220e5b1" -r 20 -e 30`


# Pre-Trained models

The Id of several very accurate pre-trained models are below, you can use these ids to load the various models in the examples

| Game          | ID                                    | GVGAI environment e.g. |
|:-------------:|:-------------------------------------:|:----------------------:|
| sokoban       | 308e9d7c-371f-495d-9f53-95170220e5b1  | gvgai-sokoban-lvl0-v0  |
