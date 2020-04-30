# Neural-Game-Engine
Code to reproduce Neural Game Engine experiments and pre-trained models


# Setup

Firstly clone the Neural Game Engine repository:

```
git clone git@github.com:Bam4d/Neural-Game-Engine.git
```

Create a conda environment using the following command:
```
conda create --name NGE python=3.7
```

Activate the environment
```
conda activate NGE
```

Install pytorch (and cuda)
```
conda install pytorch torchvision tensorboard cudatoolkit=10.1 -c pytorch
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

### Level Generation

Forward models are trained using data from generated game levels. For many games, level generation parameters have to be optimized to 
produce a good distribution of states and actions when a random agent is used.
Level generation configurations are stored in `training/environment/level_generator_configs.py`
If there is no config for a particular game in this file. 
The level generator will create statistics based on the default GVGAI levels. 
In most cases this does not produce levels that train particularly fast and sometimes are quite innaccurate.

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

The Id of several accurate pre-trained models are below, you can use these ids to load the various models in the examples

| Game          | ID                                    | GVGAI environment example   |
|:-------------:|:-------------------------------------:|:---------------------------:|
| sokoban       | 4e899a4f-80f4-4abe-8f80-87ae6f1a610c  | gvgai-sokoban-lvl0-v0       |
| cookmepasta   | 156d9619-cea9-44ab-80e4-7d7ef937f094  | gvgai-cookmepasta-lvl0-v0   |
| bait          | b59b1802-6285-44c5-9ee5-4e12f8da8a5c  | gvgai-bait-lvl0-v0          |
| brainman      | 595c1e30-8af6-4383-bb6f-68efd652d29f  | gvgai-brainman-lvl0-v0      |
| labyrinth     | 7752dbb5-facf-4b3a-934a-2c1e69fad6be  | gvgai-labyrinth-lvl0-v0     |
| realsokoban   | 1a9a5d6f-013e-44d0-9c77-f30b633bdd54  | gvgai-realsokoban-lvl0-v0   |
| painter       | 59c7810f-7112-4a3a-bf65-b6fcb8718ad5  | gvgai-painter-lvl1-v0       |
| zenpuzzle     | 21654615-5656-404e-9aec-1a16a13dab47  | gvgai-zenpuzzle-lvl0-v0     |
| clusters      | 9ccce85b-5497-4377-84f0-cbe8e8ff54fd  | gvgai-clusters-lvl0-v0      |
| aliens        | 6c6b2bb3-f9f6-4b0a-8f15-19269b05acc0  | gvgai-aliens-lvl0-v0        |