# Some optimized level generator configs that work well when producting data for training games.
# Ideally, these are used to tune the reward rate metric to match the 'reward_class_weight'.
# For example, if the reward rate is roughly 0.01 (1 reward per 100 frames) then 'reward_class_weight' should be [0.01, 0.99]
# The average reward rate is plotted to tensorboard and also is seen in the console during training
level_generator_configs = {
    'sokoban': [{
        'min_height': 8,
        'min_width': 8,
        'max_height': 10,
        'max_width': 10,
        'tiles': {
            'w': {'types': ['edge', 'sparse'], 'prob': 0.1},
            '.': {'types': ['sparse'], 'prob': 0.6},
            '1': {'types': ['sparse'], 'prob': 0.15},
            'A': {'types': ['singleton']},
            '0': {'types': ['sparse'], 'prob': 0.15}
        }
    }],

}
