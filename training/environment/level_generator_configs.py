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
    'cookmepasta': [{
        'min_width': 5,
        'min_height': 5,
        'max_width': 7,
        'max_height': 7,
        'tiles': {
            'w': {'types': ['edge', 'sparse'], 'prob': 0.15},
            '.': {'types': ['sparse'], 'prob': 0.85},
            'p': {'types': ['singleton']},
            'o': {'types': ['singleton']},
            'A': {'types': ['singleton']},
            'b': {'types': ['singleton']},
            't': {'types': ['singleton']},
            'l': {'types': ['sparse'], 'prob': 0.00},
            'k': {'types': ['sparse'], 'prob': 0.00}
        }
    }],
    "bait": [{
        'min_width': 8.0,
        'min_height': 8.0,
        'max_width': 12.0,
        'max_height': 12.0,
        'tiles': {
            'w': {'types': ['edge', 'sparse'], 'prob': 0.1},
            'g': {'types': ['singleton']},
            'A': {'types': ['singleton']},
            '.': {'types': ['sparse'], 'prob': 0.5},
            '1': {'types': ['sparse'], 'prob': 0.2},
            'k': {'types': ['singleton']},
            '0': {'types': ['sparse'], 'prob': 0.2},
            'm': {'types': ['sparse'], 'prob': 0.00}
        }
    }],
    "brainman": [{
        'min_width': 8.0,
        'min_height': 8.0,
        'max_width': 12.0,
        'max_height': 12.0,
        'tiles': {
            'w': {'types': ['edge', 'sparse'], 'prob': 0.1},
            'A': {'types': ['singleton']},
            '.': {'types': ['sparse'], 'prob': 0.6},
            'g': {'types': ['sparse'], 'prob': 0.05},
            'r': {'types': ['sparse'], 'prob': 0.05},
            'b': {'types': ['sparse'], 'prob': 0.05},
            'd': {'types': ['sparse'], 'prob': 0.05},
            'e': {'types': ['singleton']},
            'k': {'types': ['sparse'], 'prob': 0.01},
            'O': {'types': ['sparse'], 'prob': 0.01}
        }
    }],
    "labyrinth": [{
        'min_width': 8,
        'max_width': 8,
        'min_height': 12,
        'max_height': 12,
        'tiles': {'w': {'types': ['edge', 'sparse'], 'prob': 0.5},
                  '.': {'types': ['sparse'], 'prob': 0.45},
                  'x': {'types': ['singleton']},
                  't': {'types': ['sparse'], 'prob': 0.05},
                  'A': {'types': ['singleton']}
                  }
    }]

}
